#!/usr/bin/env bash
# tools/e2e/soak.sh — task 4.4: the soak test of the live bot against the LOCAL DDNet server (127.0.0.1:8303).
#
#   tools/e2e/soak.sh [--build] <soak.py options>
#   tools/e2e/soak.sh --build --label A-wbauto                      # 60 min, WB auto, bot as a child process, console fed
#   tools/e2e/soak.sh --label B-wboff --wb off --unit               # 60 min, WB off, through deploy/systemd/ddnet-ai-bot.service
#   tools/e2e/soak.sh --label rehearsal --duration 600              # a 10 minute rehearsal (the whole timeline, compressed)
#   tools/e2e/soak.sh --label rehearsal-6h --real-data --duration 21600 --out ~/aiddnet/data/logs/4.5   # task 4.5: the production layout
#                                                                  # (unmodified unit, REAL ~/aiddnet/data/bot; never moves or deletes anything there)
#   tools/e2e/soak.sh --label rehearsal-fly --real-data --duration 1500 --fly-bundle <bundle> --out ...   # the same with hybrid:fly
#   tools/e2e/soak.sh --label kfb --unit --kill-protection 2 --duration 2400   # task 4.6: sv_kill_protection 2 for the run (restored to 20, read back), the /kill fallback exercised
#   tools/e2e/soak.sh --analyze ~/aiddnet/data/logs/4.4/<run>       # re-run the analysis of a finished run
#   tools/e2e/soak.sh --selftest                                    # the analysis can fail (synthetic runs)
#
# Needs: the local server (ddnet-local.service, you own it for the duration), passwordless sudo for `systemctl restart
# ddnet-local.service` (and, with --unit, for installing the bot unit), python3. Builds with <= 3 jobs. Loopback only: the
# harness never touches another address, never the production web on 7788 (the web unit of the soak is on --web-port,
# default 7790) and never ~/aiddnet/bin. Journal and report: ~/aiddnet/data/logs/4.4/<stamp>-<label>/ (not in git).
# Restores the server's map to "Copy Love Box" at the end and reads it back (also after Ctrl-C).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

case "${1:-}" in
    --selftest) python3 "$HERE/soak.py" --selftest && exec python3 "$HERE/soak_analyze.py" --selftest ;;
    --analyze) shift; exec python3 "$HERE/soak_analyze.py" "$@" ;;
esac

BUILD=0
if [ "${1:-}" = "--build" ]; then
    BUILD=1
    shift
fi
if [ "$BUILD" = 1 ] || [ ! -x "$REPO_ROOT/target/release/ddnet-ai" ]; then
    # shellcheck disable=SC1090,SC1091
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    echo "building ddnet-ai (release, 3 jobs) ..."
    (cd "$REPO_ROOT" && CARGO_BUILD_JOBS=3 cargo build --release --locked -p ddnet-ai)
fi
exec python3 "$HERE/soak.py" "$@"
