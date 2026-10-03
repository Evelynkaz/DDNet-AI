#!/usr/bin/env python3
"""Task 4.5: does DDNet's `sv_kill_protection` make the bot's `Cl_Kill` do nothing after 20 minutes of a life?

Evidence from the 6 h rehearsal: the bot froze itself 22.8 minutes into a life, sent `Cl_Kill` every 10 s for 93 minutes and was never
killed (docs/EXPERIMENTS.md E-011). DDNet 20.1 `CGameContext::OnKillNetMessage` (~/aiddnet/build/ddnet-20.1/src/src/game/server/gamecontext.cpp:2977; `Cl_SetTeam`, line 2701, is
protected the same way, so going to the spectators and back is no way out) drops the kill silently
(a chat line to the player, which the bot never reads) when `sv_kill_protection != 0`, the life is older than that many minutes and the
race state is STARTED. This probe reproduces it on the LOCAL server: it sets `sv_kill_protection 1` (one minute, restored to the value it
had at the end), lets a bot play for 75 s without dying, sends the console `!kill` and sees whether a new life starts; then the same with
`sv_kill_protection 0`. Loopback only, one bot, the server must be empty (it is yours for the duration).

    tools/e2e/kill_protection_probe.py [--bin <ddnet-ai>] [--trials 2]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soak  # noqa: E402


def econ(*args):
    r = soak.sh([sys.executable, str(soak.ECON), *args], timeout=60)
    return r.stdout


def current(var):
    m = re.search(r"Value: (.+)", econ(var))
    return m.group(1).strip() if m else None


def trial(binary, protection):
    """One bot, `!kill` after 75 s of an unbroken life; True when the kill took effect (a new life started within 6 s)."""
    econ("sv_kill_protection", str(protection))
    data = Path(tempfile.mkdtemp(prefix="kp-probe-"))
    (data / "maps").symlink_to(soak.DATA / "maps")
    cmd = [binary, "play", "--server", soak.SERVER_ADDR, "--name", "kp-probe", "--brain", "hybrid", "--duration", "400", "--data-dir", str(data),
           "--no-bridge", "--no-control", "--no-memory", "--no-settings", "--no-autoclip", "--console"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         env=dict(os.environ, **soak.BOT_ENV), start_new_session=True)
    lives = []  # monotonic times of "life started"
    lock = threading.Lock()

    def tail():
        for raw in iter(p.stderr.readline, b""):
            if b"life started" in raw:
                with lock:
                    lives.append(time.monotonic())

    threading.Thread(target=tail, daemon=True).start()
    try:
        # wait for a life that has lasted 75 s without a new one (the bot may die and respawn meanwhile), at most 5 minutes
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            with lock:
                last = lives[-1] if lives else None
            if last and time.monotonic() - last >= 75:
                break
            time.sleep(1)
        else:
            return None
        with lock:
            before = len(lives)
        p.stdin.write(b"!kill\n")
        p.stdin.flush()
        time.sleep(6)
        with lock:
            return len(lives) > before
    finally:
        try:
            p.stdin.write(b"!quit\n")
            p.stdin.flush()
            p.wait(20)
        except (OSError, subprocess.TimeoutExpired):
            p.kill()
        shutil.rmtree(data, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=str(soak.REPO / "target" / "release" / "ddnet-ai"))
    ap.add_argument("--trials", type=int, default=2)
    a = ap.parse_args()
    if soak.re.search(r"name='", econ("status")):
        raise SystemExit("someone is on the local server: refusing")
    original = current("sv_kill_protection")
    print("sv_kill_protection was", original)
    results = {}
    try:
        for value in (1, 0):
            results[value] = [trial(a.bin, value) for _ in range(a.trials)]
            print(f"sv_kill_protection {value}: kill took effect after 75 s of life -> {results[value]}", flush=True)
    finally:
        if original is not None:
            econ("sv_kill_protection", original)
        print("sv_kill_protection restored to", current("sv_kill_protection"))
    ok = all(r is False for r in results[1]) and all(r is True for r in results[0])
    print("REPRODUCED" if ok else "NOT REPRODUCED (or a trial timed out)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
