//! Task 2.3's e2e ask ("an `#[ignore]` Rust integration test that runs it when `DDAI_E2E=1`"):
//! runs `tools/e2e/session.sh` (every scenario a-h/d2) against the real local DDNet 20.1 server
//! (`ddnet-local.service`, 127.0.0.1:8303) when explicitly requested. Never run by a plain `cargo
//! test`/`cargo test --workspace` (this project's live-play policy — 127.0.0.1-only,
//! rate-limited connections, no auto-reconnect-after-kick/ban, restores the server to "Copy Love
//! Box" on exit — is meant to be exercised deliberately, not incidentally by CI or routine local
//! development): `#[ignore]` keeps it out of a default run, and the `DDAI_E2E=1` check below is a
//! second, independent guard in case someone runs `cargo test -- --ignored` without meaning to
//! touch a real, shared local server.
//!
//! Requires: `ddnet-local.service` running, `sudo` access for `systemctl`/`kill` (the script's own
//! scenarios (d)/(d2) need it), and `python3` (`tools/e2e/analyze_positions.py`,
//! `tools/ddnet-server/econ.py`). See `tools/e2e/session.sh`'s own doc comment for what each
//! scenario does; see `~/aiddnet/data/logs/e2e-2.3/<timestamp>/` for that run's own logs
//! afterwards regardless of pass/fail.

use std::env;
use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "touches the real local ddnet-local.service; run explicitly with DDAI_E2E=1"]
fn e2e_session_against_the_local_server() {
    if env::var_os("DDAI_E2E").as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!(
            "skipping: this test touches the real local ddnet-local.service — set DDAI_E2E=1 to \
             actually run tools/e2e/session.sh"
        );
        return;
    }

    // `crates/ddai-client` -> `crates` -> the repo root.
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ddai-client crate is two levels under the repo root")
        .to_path_buf();
    let script = repo_root.join("tools").join("e2e").join("session.sh");
    assert!(
        script.is_file(),
        "expected {} to exist (repo root resolved to {})",
        script.display(),
        repo_root.display()
    );

    let status = Command::new("bash")
        .arg(&script)
        .current_dir(&repo_root)
        .status()
        .expect("failed to spawn tools/e2e/session.sh");
    assert!(
        status.success(),
        "tools/e2e/session.sh exited with {status:?} — see \
         ~/aiddnet/data/logs/e2e-2.3/<timestamp>/ for that run's own per-scenario logs"
    );
}
