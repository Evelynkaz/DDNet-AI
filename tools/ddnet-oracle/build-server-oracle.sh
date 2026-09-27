#!/usr/bin/env bash
# Builds Oracle B: fetches a COPY of the pinned DDNet 20.1 source tree (same tag/commit as
# tools/ddnet-server/build.sh and tools/ddnet-oracle/fetch.sh -- see EXPECTED_COMMIT below),
# applies a small CMake overlay (server/cmake-overlay.cmake) that adds the `ddai_oracle_server`
# executable target, and builds ONLY that target (not the full DDNet-Server) with the same
# CMake configuration (-DCLIENT=OFF -DTOOLS=OFF -DDOWNLOAD_GTEST=OFF) task 2.1's
# tools/ddnet-server/build.sh uses for the real server.
#
# This is deliberately a SEPARATE checkout+build directory from both
# tools/ddnet-server/build.sh's ($DDNET_BUILD_ROOT, default ~/aiddnet/build/ddnet-20.1 -- the
# tree the running ddnet-local.service was built from) and tools/ddnet-oracle/fetch.sh's
# (tools/ddnet-oracle/build/ddnet-src, Oracle A's core-only checkout): Oracle B never reads from
# or writes into either of those, and never touches the running local server.
#
# Usage: tools/ddnet-oracle/build-server-oracle.sh
#
# Env overrides (all optional):
#   ORACLE_B_BUILD_ROOT   default: $HOME/aiddnet/build/oracle-b
#   DDNET_TAG             default: 20.1
#   DDNET_COMMIT          default: c9d208138f85755521f16a0096b6fe036c5c8698
#   DDNET_REPO_URL        default: https://github.com/ddnet/ddnet.git
#   ORACLE_B_BUILD_JOBS   default: 4   (bounded, per project convention -- CLAUDE.md)
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

ORACLE_B_BUILD_ROOT="${ORACLE_B_BUILD_ROOT:-$HOME/aiddnet/build/oracle-b}"
DDNET_TAG="${DDNET_TAG:-20.1}"
DDNET_COMMIT="${DDNET_COMMIT:-c9d208138f85755521f16a0096b6fe036c5c8698}"
DDNET_REPO_URL="${DDNET_REPO_URL:-https://github.com/ddnet/ddnet.git}"
ORACLE_B_BUILD_JOBS="${ORACLE_B_BUILD_JOBS:-4}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

SRC_DIR="$ORACLE_B_BUILD_ROOT/src"
BIN_DIR="$ORACLE_B_BUILD_ROOT/build"
OVERLAY_FILE="$SCRIPT_DIR/server/cmake-overlay.cmake"
HARNESS_SRC="$SCRIPT_DIR/server/oracle_server.cpp"
MARKER="ddai_oracle_server"
ANCHOR='add_custom_target(everything DEPENDS ${TARGETS_OWN})'

log() { printf '[build-server-oracle.sh] %s\n' "$*" >&2; }
die() { printf '[build-server-oracle.sh] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "$OVERLAY_FILE" ]] || die "$OVERLAY_FILE not found"
[[ -f "$HARNESS_SRC" ]] || die "$HARNESS_SRC not found"

# ---------------------------------------------------------------------------
# 1. Fetch (shallow, pinned-tag-verified) if not already present with the overlay applied.
# ---------------------------------------------------------------------------
already_patched=0
if [[ -d "$SRC_DIR/.git" ]]; then
  actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD)"
  if [[ "$actual_commit" == "$DDNET_COMMIT" ]] && grep -qF "$MARKER" "$SRC_DIR/CMakeLists.txt"; then
    log "$SRC_DIR already at $DDNET_COMMIT with the overlay applied, reusing"
    already_patched=1
  else
    log "$SRC_DIR is at $actual_commit (or missing the overlay) -- re-fetching from scratch"
    rm -rf "$SRC_DIR"
  fi
fi

if [[ ! -d "$SRC_DIR/.git" ]]; then
  mkdir -p "$ORACLE_B_BUILD_ROOT"
  log "cloning $DDNET_REPO_URL @ $DDNET_TAG -> $SRC_DIR (shallow)"
  git clone --quiet --depth 1 --branch "$DDNET_TAG" "$DDNET_REPO_URL" "$SRC_DIR"
  actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD)"
  if [[ "$actual_commit" != "$DDNET_COMMIT" ]]; then
    rm -rf "$SRC_DIR"
    die "tag $DDNET_TAG resolved to $actual_commit, expected $DDNET_COMMIT (refusing to build -- possible tag tampering or wrong repo)"
  fi
  log "verified DDNet $DDNET_TAG == commit $actual_commit"
fi

# ---------------------------------------------------------------------------
# 2. Apply the CMake overlay (idempotent: only if the marker isn't already there).
# ---------------------------------------------------------------------------
if [[ "$already_patched" -ne 1 ]]; then
  log "applying CMake overlay ($OVERLAY_FILE) before '$ANCHOR'"
  python3 - "$SRC_DIR/CMakeLists.txt" "$OVERLAY_FILE" "$ANCHOR" <<'PYEOF'
import sys
cmakelists_path, overlay_path, anchor = sys.argv[1:4]
with open(cmakelists_path, "r", encoding="utf-8") as f:
    text = f.read()
with open(overlay_path, "r", encoding="utf-8") as f:
    overlay = f.read()
idx = text.find(anchor)
if idx < 0:
    sys.exit(f"anchor line not found in {cmakelists_path}: {anchor!r}")
patched = text[:idx] + overlay + "\n" + text[idx:]
with open(cmakelists_path, "w", encoding="utf-8") as f:
    f.write(patched)
PYEOF
  grep -qF "$MARKER" "$SRC_DIR/CMakeLists.txt" || die "overlay apply appears to have failed (marker not found after patching)"
  log "overlay applied"
fi

# ---------------------------------------------------------------------------
# 3. Rust toolchain (engine_shared is a Rust static lib linked into every server target,
#    including ours -- same requirement as tools/ddnet-server/build.sh).
# ---------------------------------------------------------------------------
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"
command -v rustc >/dev/null 2>&1 || die "rustc not found on PATH even after sourcing ~/.cargo/env"
command -v cargo >/dev/null 2>&1 || die "cargo not found on PATH even after sourcing ~/.cargo/env"
log "using $(rustc --version) / $(cargo --version), CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS"

# ---------------------------------------------------------------------------
# 4. Configure: server-only Release build (same flags as tools/ddnet-server/build.sh), plus
#    DDAI_ORACLE_SERVER_SRC pointing at our own harness source file (see cmake-overlay.cmake).
# ---------------------------------------------------------------------------
log "configuring cmake (Release, CLIENT=OFF, TOOLS=OFF) -> $BIN_DIR"
cmake -S "$SRC_DIR" -B "$BIN_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLIENT=OFF \
  -DTOOLS=OFF \
  -DDOWNLOAD_GTEST=OFF \
  -DDDAI_ORACLE_SERVER_SRC="$HARNESS_SRC"

# ---------------------------------------------------------------------------
# 5. Build only our target -- not the full DDNet-Server (that binary is task 2.1's, and this
#    script must never touch it).
# ---------------------------------------------------------------------------
log "building target 'ddai_oracle_server' with -j$ORACLE_B_BUILD_JOBS"
START_TS=$(date +%s)
cmake --build "$BIN_DIR" -j"$ORACLE_B_BUILD_JOBS" --target ddai_oracle_server
END_TS=$(date +%s)
log "build took $((END_TS - START_TS))s"

BINARY="$BIN_DIR/ddai_oracle_server"
[[ -x "$BINARY" ]] || die "build finished but $BINARY is missing/not executable"
log "built $BINARY ($(du -h "$BINARY" | cut -f1))"

# ---------------------------------------------------------------------------
# 6. Report the effective compiler flags + confirm no FMA contraction (acceptance criterion 1).
# ---------------------------------------------------------------------------
GAMECORE_OBJ=$(find "$BIN_DIR" -name 'gamecore.cpp.o' | head -1)
if [[ -n "$GAMECORE_OBJ" ]]; then
  FMA_COUNT=$(objdump -d "$GAMECORE_OBJ" 2>/dev/null | grep -c vfmadd || true)
  log "objdump -d $GAMECORE_OBJ | grep -c vfmadd = $FMA_COUNT (must be 0)"
  [[ "$FMA_COUNT" -eq 0 ]] || die "FMA contraction detected in gamecore.cpp.o ($FMA_COUNT vfmadd instructions) -- float parity with the Rust port would not hold"
else
  log "WARNING: could not find gamecore.cpp.o under $BIN_DIR to check for FMA contraction"
fi
ninja -C "$BIN_DIR" -t commands ddai_oracle_server 2>/dev/null | grep -m1 'gamecore\.cpp\.o' | tee "$ORACLE_B_BUILD_ROOT/last-gamecore-compile-command.txt" >&2 || true
