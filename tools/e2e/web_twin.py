#!/usr/bin/env python3
"""Task 4.5: a TWIN of the production web unit, for checking what the production unit cannot be asked to do.

`start`: a transient systemd unit `ddai-web-twin` on 127.0.0.1:7791 with the production unit's own sandbox (every hardening property is read
from deploy/systemd/ddnet-ai-web.service) and its own private data dir (own password, own session key: the production secrets are never
read or written), pointed at the REAL bot's `data/bot` (bridge, control socket, relations.json), with `data/bot` as its only other writable
path, freshly bind-mounted. Why: a unit's ReadWritePaths are bind-mounted when the unit starts, and a `data/bot` that was moved or recreated
afterwards leaves the old mount behind (read-only view of the new directory: "Не удалось записать файл списков"). The twin shows what the
production unit does after a restart. `stop` removes it. The password file is `<run dir>/secrets/web-password.txt` (0600); it never goes to stdout.

    tools/e2e/web_twin.py start [--dir ~/aiddnet/data/scratch/task45/twin] [--bin <ddnet-ai>] [--port 7791]
    tools/e2e/web_twin.py stop
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soak  # noqa: E402

UNIT = "ddai-web-twin"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["start", "stop"])
    ap.add_argument("--dir", default=str(soak.DATA / "scratch" / "task45" / "twin"))
    ap.add_argument("--bin", default=str(soak.REPO / "target" / "release" / "ddnet-ai"))
    ap.add_argument("--port", type=int, default=7791)
    a = ap.parse_args()
    if a.action == "stop":
        soak.sh(["sudo", "systemctl", "stop", UNIT])
        return
    if a.port in (soak.PROD_WEB_PORT, 7790):
        raise SystemExit("refusing: that port belongs to the production web or the soak's own web")
    run = Path(a.dir)
    (run / "secrets").mkdir(parents=True, exist_ok=True)
    os.chmod(run / "secrets", 0o700)
    soak.sh([a.bin, "web-passwd", "--data-dir", str(run)], check=True)
    bot = soak.DATA / "bot"
    props = soak.web_sandbox_props(soak.REPO / "deploy" / "systemd" / "ddnet-ai-web.service", f"{run} {bot}")
    cmd = [a.bin, "web", "--listen", f"127.0.0.1:{a.port}", "--data-dir", str(run), "--bot-socket", str(bot / "live.sock"),
           "--control-socket", str(bot / "control.sock"), "--relations", str(bot / "relations.json"),
           "--maps-dir", str(soak.DATA / "maps" / "cache")]
    soak.sh(["sudo", "systemctl", "stop", UNIT])
    soak.sh(["sudo", "systemd-run", "--unit", UNIT, "--collect", "--quiet", "-p", "User=ubuntu", "-p", "Group=ubuntu",
             "-p", f"WorkingDirectory={run}", "--setenv=NO_COLOR=1", *props, *cmd], check=True)
    for _ in range(40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", a.port)) == 0:
                print(f"twin up on http://127.0.0.1:{a.port}; password file {run / 'secrets' / 'web-password.txt'}")
                return
        time.sleep(0.5)
    raise SystemExit("the twin did not come up")


if __name__ == "__main__":
    main()
