# DDNet-AI Oracle B (task 1.5): additional executable target that reuses the SAME object
# libraries as DDNet's own `game-server` executable (game-server-without-main + engine-shared +
# game-shared + rust-bridge-shared, see DDNet's CMakeLists.txt around its own
# `add_executable(game-server ...)`), just with our own main() (oracle_server.cpp) instead of
# src/engine/server/main.cpp.
#
# Our own harness source never lives inside the (fetched, never-committed) DDNet source tree --
# it stays in this repository (tools/ddnet-oracle/server/oracle_server.cpp) and is passed in via
# the DDAI_ORACLE_SERVER_SRC cache variable, set on the `cmake -S ... -B ...` command line by
# ../build-server-oracle.sh's `apply_overlay` function, which appends this file's contents,
# verbatim, to a COPY of DDNet's own top-level CMakeLists.txt (never to the original; that copy
# lives outside this repository and is git-ignored/regenerated from scratch by that script's
# fetch step, same as tools/ddnet-server/build.sh and tools/ddnet-oracle/fetch.sh already do for
# their own DDNet checkouts). This is therefore never part of upstream DDNet.
if(DDAI_ORACLE_SERVER_SRC AND SERVER)
  add_executable(ddai_oracle_server
    ${DEPS}
    "${DDAI_ORACLE_SERVER_SRC}"
    $<TARGET_OBJECTS:game-server-without-main>
    $<TARGET_OBJECTS:engine-shared>
    $<TARGET_OBJECTS:game-shared>
    $<TARGET_OBJECTS:rust-bridge-shared>
  )
  target_link_libraries(ddai_oracle_server ${LIBS_SERVER})
  list(APPEND TARGETS_OWN ddai_oracle_server)
endif()
