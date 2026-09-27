#!/usr/bin/env bash
# Builds Oracle A: compiles the real DDNet 20.1 gamecore/collision/layers/teamscore objects
# (fetched by fetch.sh into build/, never committed — see build/ddnet-oracle/.gitignore's
# parent rule in the repo root .gitignore) together with this tool's own oracle_core.cpp, into
# build/oracle_core.
#
# Usage: ./fetch.sh && ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

readonly SRC="build/ddnet-src/src"
readonly GEN="build/generated"

if [ ! -d "$SRC" ]; then
	echo "build.sh: $SRC not found — run ./fetch.sh first" >&2
	exit 1
fi
if [ ! -f "$GEN/protocol.h" ]; then
	echo "build.sh: $GEN/protocol.h not found — run ./fetch.sh first" >&2
	exit 1
fi

mkdir -p build/generated # generated/ must be a subdirectory of an include path (`#include
# <generated/protocol.h>`), so build/generated (already created by fetch.sh) is included via
# `-I build`, not `-I build/generated`.

# Per the task spec: no `-march` (keeps float codegen portable — no FMA contraction — see
# docs/research/ddnet-physics.md §4), `-fno-fast-math`/`-ffp-contract=off` (no reassociation, no
# contraction even if a future compiler defaults `-ffp-contract=fast` without `-march`),
# `-fsigned-char` (matches DDNet's own build: `char` is signed on DDNet's target platforms, and
# some tile-index comparisons are sign-sensitive).
readonly CXXFLAGS="-std=c++20 -O2 -fno-fast-math -ffp-contract=off -fsigned-char -I $SRC -I build"

readonly DDNET_SOURCES="game/gamecore.cpp game/collision.cpp game/layers.cpp game/teamscore.cpp game/prng.cpp"
mkdir -p build/obj
OBJS=()
for src in $DDNET_SOURCES; do
	obj="build/obj/$(basename "$src" .cpp).o"
	echo "build.sh: g++ -c $SRC/$src"
	g++ $CXXFLAGS -c "$SRC/$src" -o "$obj"
	OBJS+=("$obj")
done

echo "build.sh: g++ -c oracle_core.cpp"
g++ $CXXFLAGS -c oracle_core.cpp -o build/obj/oracle_core.o

echo "build.sh: linking build/oracle_core"
g++ $CXXFLAGS build/obj/oracle_core.o "${OBJS[@]}" -o build/oracle_core

echo "build.sh: built build/oracle_core"

# --- map2raw (task 1.4): real datafile/map/layers loading, not the core-only harness above. ----
# Needs the same game/*.cpp objects as oracle_core (layers.cpp for CLayers; gamecore/collision/
# teamscore/prng.cpp only because layers.cpp's link set pulls them in transitively — see
# map2raw.cpp's header comment) plus the real datafile/map readers and their base/ dependencies.
# `-lz` links the system zlib (DDNet's own `<zlib.h>` usage in datafile.cpp) — see
# map2raw.cpp's header comment and docs/formats.md for why the Rust side uses a pure-Rust zlib
# instead (D-009-adjacent: keep native deps out of the crate we ship, not the disposable oracle).
readonly MAP2RAW_DDNET_SOURCES="engine/shared/datafile.cpp engine/shared/map.cpp game/layers.cpp game/gamecore.cpp game/collision.cpp game/teamscore.cpp game/prng.cpp base/str.cpp base/mem.cpp base/io.cpp base/hash_libtomcrypt.cpp base/bytes.cpp base/unicode/tolower.cpp base/unicode/tolower_data.cpp"
MAP2RAW_OBJS=()
for src in $MAP2RAW_DDNET_SOURCES; do
	obj="build/obj/map2raw_$(basename "$src" .cpp).o"
	echo "build.sh: g++ -c $SRC/$src (map2raw)"
	g++ $CXXFLAGS -c "$SRC/$src" -o "$obj"
	MAP2RAW_OBJS+=("$obj")
done

echo "build.sh: g++ -c map2raw.cpp"
g++ $CXXFLAGS -c map2raw.cpp -o build/obj/map2raw.o

echo "build.sh: linking build/map2raw"
g++ $CXXFLAGS build/obj/map2raw.o "${MAP2RAW_OBJS[@]}" -lz -o build/map2raw

echo "build.sh: built build/map2raw"
