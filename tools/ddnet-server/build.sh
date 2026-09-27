#!/usr/bin/env bash
# Build the DDNet dedicated server (version 20.1) from source for local, non-public use.
#
# What it does:
#   1. Installs the apt packages the 20.1 SERVER-ONLY build needs (see PACKAGES below).
#   2. Clones (or updates) DDNet at tag 20.1 into a source dir OUTSIDE this repo and
#      verifies it resolved to the exact commit the spec pins.
#   3. Configures CMake for a server-only Release build (-DCLIENT=OFF -DTOOLS=OFF).
#   4. Builds the `DDNet-Server` target with a bounded job count, sampling memory
#      use while the build runs, and prints a summary (incl. peak RSS) at the end.
#
# Re-running is safe/idempotent: cloning is skipped if the source dir already exists,
# checkout of the pinned tag is a no-op if already checked out, `cmake` re-configure
# is cheap, and `ninja`/`cmake --build` only rebuilds what changed.
#
# Env overrides (all optional):
#   DDNET_BUILD_ROOT   default: $HOME/aiddnet/build/ddnet-20.1
#   DDNET_TAG          default: 20.1
#   DDNET_COMMIT       default: c9d208138f85755521f16a0096b6fe036c5c8698
#   DDNET_REPO_URL     default: https://github.com/ddnet/ddnet.git
#   DDNET_BUILD_JOBS   default: 4          (bounded; see README.md "Сборка" for the measured
#                                            peak RSS this produced and whether nproc is safe)
#   DDNET_SKIP_APT     default: unset      (set to 1 to skip the `apt-get install` step,
#                                            e.g. if packages are already provisioned
#                                            some other way, or sudo is unavailable)
#
# Usage: tools/ddnet-server/build.sh
set -euo pipefail

DDNET_BUILD_ROOT="${DDNET_BUILD_ROOT:-$HOME/aiddnet/build/ddnet-20.1}"
DDNET_TAG="${DDNET_TAG:-20.1}"
DDNET_COMMIT="${DDNET_COMMIT:-c9d208138f85755521f16a0096b6fe036c5c8698}"
DDNET_REPO_URL="${DDNET_REPO_URL:-https://github.com/ddnet/ddnet.git}"
DDNET_BUILD_JOBS="${DDNET_BUILD_JOBS:-4}"

SRC_DIR="$DDNET_BUILD_ROOT/src"
BIN_DIR="$DDNET_BUILD_ROOT/build"
SERVER_TARGET="DDNet-Server"

log() { printf '[build.sh] %s\n' "$*" >&2; }
die() { printf '[build.sh] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. System packages needed for a SERVER-ONLY (CLIENT=OFF, TOOLS=OFF) 20.1
#    build. Determined from ddnet/CMakeLists.txt @ 20.1 (c9d20813...):
#      - find_package(Curl)/(SQLite3) are unconditionally required
#        (message(SEND_ERROR ...) if missing, regardless of CLIENT/TOOLS).
#      - find_package(Freetype)/(Ogg)/(Opus)/(Opusfile)/(SDL2)/(PNG)/(GLEW)/
#        (FFMPEG)/(Vulkan) are only required "if(CLIENT ...)" / "if(CLIENT OR
#        TOOLS)" - NOT needed for CLIENT=OFF TOOLS=OFF.
#      - find_package(Rust) is unconditionally required (engine_shared is a
#        Rust static lib linked into every target, client or server).
#      - Zlib/OpenSSL(Crypto) are linked into the server too (network/demo
#        compression, hashing) - confirmed by an actual configure+link below.
#    Verified empirically: `cmake -DCLIENT=OFF -DTOOLS=OFF -DDOWNLOAD_GTEST=OFF`
#    configures and `DDNet-Server` links cleanly with exactly this set, on a
#    stock Ubuntu 24.04 (nothing more, nothing less).
# ---------------------------------------------------------------------------
PACKAGES=(
  build-essential   # gcc/g++/make - C++17 compiler toolchain
  cmake             # build system generator DDNet uses
  ninja-build       # fast backend for cmake -G Ninja
  pkg-config        # used by FindCurl/FindZLIB/etc via pkg_check_modules
  git               # `git describe`, submodule-free here but cmake checks for it
  python3           # datasrc/*.py code generators (protocol, content types)
  libcurl4-openssl-dev # engine/shared/http, map/asset downloads (Curl, required)
  libsqlite3-dev    # engine/server/databases/sqlite.cpp (SQLite3, required)
  libssl-dev        # OpenSSL Crypto (hashing; avoids the bundled md5 fallback)
  zlib1g-dev         # zlib (network/demo compression; linked into the server)
)

if [[ "${DDNET_SKIP_APT:-0}" != "1" ]]; then
  log "installing/verifying apt packages: ${PACKAGES[*]}"
  sudo apt-get update -y
  sudo apt-get install -y "${PACKAGES[@]}"
else
  log "DDNET_SKIP_APT=1, skipping apt-get install"
fi

# ---------------------------------------------------------------------------
# 2. Rust/cargo: DDNet 20.x links a Rust static lib (engine_shared) into the
#    server via cmake's `cargo build --locked`. Use the rustup toolchain
#    already installed at ~/.cargo (MSRV per ddnet 20.1 cmake/FindRust.cmake
#    is 1.85.0) - do NOT install another Rust.
# ---------------------------------------------------------------------------
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"

command -v rustc >/dev/null 2>&1 || die "rustc not found on PATH even after sourcing ~/.cargo/env"
command -v cargo >/dev/null 2>&1 || die "cargo not found on PATH even after sourcing ~/.cargo/env"
log "using $(rustc --version) / $(cargo --version)"

# ---------------------------------------------------------------------------
# 3. Obtain DDNet at the pinned tag, OUTSIDE the repo, and verify the commit.
# ---------------------------------------------------------------------------
mkdir -p "$DDNET_BUILD_ROOT"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  log "cloning $DDNET_REPO_URL -> $SRC_DIR"
  git clone --quiet "$DDNET_REPO_URL" "$SRC_DIR"
else
  log "source dir already exists at $SRC_DIR, reusing (fetching tags)"
fi

git -C "$SRC_DIR" fetch --quiet --tags origin

# Verify the tag itself points at the expected commit before trusting it.
TAG_COMMIT="$(git -C "$SRC_DIR" rev-parse "refs/tags/$DDNET_TAG^{commit}")"
if [[ "$TAG_COMMIT" != "$DDNET_COMMIT" ]]; then
  die "tag $DDNET_TAG resolved to $TAG_COMMIT, expected $DDNET_COMMIT (refusing to build - possible tag tampering or wrong repo)"
fi

if [[ -n "$(git -C "$SRC_DIR" status --porcelain)" ]]; then
  die "$SRC_DIR has local modifications; refusing to check out over them. Clean it up or point DDNET_BUILD_ROOT elsewhere."
fi

git -C "$SRC_DIR" checkout --quiet "$DDNET_TAG"

ACTUAL_COMMIT="$(git -C "$SRC_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$DDNET_COMMIT" ]]; then
  die "checked out HEAD is $ACTUAL_COMMIT, expected $DDNET_COMMIT"
fi
log "verified DDNet $DDNET_TAG == commit $ACTUAL_COMMIT"

# ---------------------------------------------------------------------------
# 4. Configure: server-only Release build. No client, no tools, no bundled
#    GTest download (we don't run the test suite here).
# ---------------------------------------------------------------------------
log "configuring cmake (Release, CLIENT=OFF, TOOLS=OFF) -> $BIN_DIR"
cmake -S "$SRC_DIR" -B "$BIN_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLIENT=OFF \
  -DTOOLS=OFF \
  -DDOWNLOAD_GTEST=OFF

# ---------------------------------------------------------------------------
# 5. Build with a bounded job count, sampling RSS of the build's own process
#    tree (compiler/linker/cargo/rustc/ninja/cmake) plus system-wide used
#    memory every 0.5s, so we can report the peak instead of guessing.
#    Every run gets its OWN timestamped summary/log (never overwritten), plus
#    a "-latest" pair that always reflects the most recent run - so a real
#    clean-build measurement doesn't get clobbered by a later no-op/incremental
#    re-run's much smaller numbers.
# ---------------------------------------------------------------------------
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
MEM_LOG="$DDNET_BUILD_ROOT/build-mem-samples-$RUN_TS.log"
MEM_SUMMARY="$DDNET_BUILD_ROOT/build-mem-summary-$RUN_TS.txt"
MEM_LOG_LATEST="$DDNET_BUILD_ROOT/build-mem-samples-latest.log"
MEM_SUMMARY_LATEST="$DDNET_BUILD_ROOT/build-mem-summary-latest.txt"
: > "$MEM_LOG"

sample_mem() {
  local peak_build_kb=0 peak_sys_kb=0 build_kb sys_kb
  while true; do
    build_kb=$(ps -eo rss,comm --no-headers 2>/dev/null | awk \
      '$2 ~ /^(cc1plus|cc1|collect2|ld|ld\.bfd|ld\.gold|ld\.lld|ninja|cmake|make|rustc|cargo|c\+\+|g\+\+|gcc|as)$/ {s+=$1} END{print s+0}')
    sys_kb=$(free -k | awk '/^Mem:/{print $3}')
    echo "$(date +%s.%N) build_rss_kb=$build_kb sys_used_kb=$sys_kb" >> "$MEM_LOG"
    (( build_kb > peak_build_kb )) && peak_build_kb=$build_kb
    (( sys_kb > peak_sys_kb )) && peak_sys_kb=$sys_kb
    printf '%s %s\n' "$peak_build_kb" "$peak_sys_kb" > "$MEM_LOG.peak"
    sleep 0.5
  done
}

sample_mem &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

log "building target '$SERVER_TARGET' with -j$DDNET_BUILD_JOBS"
START_TS=$(date +%s)
cmake --build "$BIN_DIR" -j"$DDNET_BUILD_JOBS" --target "$SERVER_TARGET"
END_TS=$(date +%s)

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
trap - EXIT

read -r PEAK_BUILD_KB PEAK_SYS_KB < "$MEM_LOG.peak"
{
  echo "jobs=$DDNET_BUILD_JOBS"
  echo "duration_s=$((END_TS - START_TS))"
  echo "peak_build_rss_mb=$((PEAK_BUILD_KB / 1024))"
  echo "peak_system_used_mb=$((PEAK_SYS_KB / 1024))"
} | tee "$MEM_SUMMARY" >&2
rm -f "$MEM_LOG.peak"
cp -f "$MEM_LOG" "$MEM_LOG_LATEST"
cp -f "$MEM_SUMMARY" "$MEM_SUMMARY_LATEST"

BINARY="$BIN_DIR/$SERVER_TARGET"
[[ -x "$BINARY" ]] || die "build finished but $BINARY is missing/not executable"

log "built $BINARY ($(du -h "$BINARY" | cut -f1))"
log "memory summary written to $MEM_SUMMARY (and $MEM_SUMMARY_LATEST)"
