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
