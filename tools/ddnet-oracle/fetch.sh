#!/usr/bin/env bash
# Fetches the exact DDNet 20.1 sources Oracle A links against, into build/ (git-ignored — see
# ../../.gitignore's `tools/ddnet-oracle/build/` rule). No DDNet source is committed to this
# repository; build.sh compiles it from this fetched, gitignored checkout every time.
#
# Usage: ./fetch.sh
set -euo pipefail
cd "$(dirname "$0")"

readonly REPO_URL="https://github.com/ddnet/ddnet.git"
readonly TAG="20.1"
# Pinned in docs/DECISIONS.md D-005 and the task spec: the last stable DDNet release as of
# writing. Verified below so a tag mutation upstream (or a network MITM) can never silently
# swap in different physics code.
readonly EXPECTED_COMMIT="c9d208138f85755521f16a0096b6fe036c5c8698"
readonly SRC_DIR="build/ddnet-src"
readonly GENERATED_DIR="build/generated"

if [ -d "$SRC_DIR/.git" ]; then
	actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD)"
	if [ "$actual_commit" = "$EXPECTED_COMMIT" ]; then
		echo "fetch.sh: $SRC_DIR already at $EXPECTED_COMMIT, skipping clone"
	else
		echo "fetch.sh: $SRC_DIR is at $actual_commit, not $EXPECTED_COMMIT — re-fetching" >&2
		rm -rf "$SRC_DIR"
	fi
fi

if [ ! -d "$SRC_DIR/.git" ]; then
	mkdir -p build
	# Shallow (depth 1: only the tag's commit, no history) + sparse (only the two directories
	# Oracle A's build actually needs: `src` for the C++ sources/headers, `datasrc` for the
	# network_header codegen script build.sh calls below) clone, per the task spec.
	git clone --depth 1 --filter=blob:none --sparse --branch "$TAG" "$REPO_URL" "$SRC_DIR"
	# Cone mode (the sparse-checkout default) only takes directory prefixes, not individual
	# files — `src` and `datasrc` are the two directories build.sh needs (DDNet's zlib license
	# text is not needed at build time; see docs/DECISIONS.md and this file's own header for the
	# licensing note).
	git -C "$SRC_DIR" sparse-checkout set src datasrc

	actual_commit="$(git -C "$SRC_DIR" rev-parse HEAD)"
	if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
		echo "fetch.sh: FATAL: tag $TAG resolved to $actual_commit, expected $EXPECTED_COMMIT" >&2
		echo "fetch.sh: refusing to build against an unexpected DDNet commit; removing $SRC_DIR" >&2
		rm -rf "$SRC_DIR"
		exit 1
	fi
	echo "fetch.sh: fetched DDNet $TAG at $actual_commit"
fi

# Generate generated/protocol.h the same way the real DDNet build does (datasrc/compile.py),
# matching the phase-0 prototype (data/research/physics-scratch/run.sh).
mkdir -p "$GENERATED_DIR"
(cd "$SRC_DIR" && python3 datasrc/compile.py network_header) >"$GENERATED_DIR/protocol.h"
echo "fetch.sh: generated $GENERATED_DIR/protocol.h ($(wc -l <"$GENERATED_DIR/protocol.h") lines)"

# task 8.4b (demo2json): `engine/shared/snapshot.cpp` itself `#include`s
# `<generated/protocol7.h>` and `<generated/protocolglue.h>` (0.7/"sixup" cross-conversion
# support inside `CSnapshotBuilder::NewItem`, snapshot.cpp:14-15,924 — dead code for this repo's
# 0.6-only demo2json, see that file's header comment, but still needed to *link*: the whole
# object file is pulled in). Generated the same way real DDNet's CMake build does (`generate_
# source7`/`generate_maps` in CMakeLists.txt), via the sparse-checked-out `datasrc/seven/` and
# `datasrc/crosscompile.py` (both already present — see the sparse-checkout set above).
(cd "$SRC_DIR" && python3 -m datasrc.seven.compile network_header) >"$GENERATED_DIR/protocol7.h"
echo "fetch.sh: generated $GENERATED_DIR/protocol7.h ($(wc -l <"$GENERATED_DIR/protocol7.h") lines)"
(cd "$SRC_DIR" && python3 datasrc/crosscompile.py map_header) >"$GENERATED_DIR/protocolglue.h"
echo "fetch.sh: generated $GENERATED_DIR/protocolglue.h ($(wc -l <"$GENERATED_DIR/protocolglue.h") lines)"
(cd "$SRC_DIR" && python3 datasrc/crosscompile.py map_source) >"$GENERATED_DIR/protocolglue_generated.cpp"
echo "fetch.sh: generated $GENERATED_DIR/protocolglue_generated.cpp ($(wc -l <"$GENERATED_DIR/protocolglue_generated.cpp") lines)"
