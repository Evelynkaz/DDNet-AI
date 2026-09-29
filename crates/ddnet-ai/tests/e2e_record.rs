//! Task 8.4a acceptance criterion 4's e2e ask ("an `#[ignore]` test gated on `DDAI_E2E=1`"): runs
//! `tools/e2e/record.sh` against the real local DDNet 20.1 server (`ddnet-local.service`,
//! 127.0.0.1:8303) when explicitly requested — same pattern as
//! `crates/ddai-client/tests/e2e_local_server.rs` (task 2.3). `#[ignore]` keeps it out of a
//! default run; the `DDAI_E2E=1` check is a second, independent guard.
//!
//! Requires: `ddnet-local.service` running, `sudo` access for `systemctl` (the script's own
//! server-restart step needs it), and `python3` (`tools/ddnet-server/econ.py`, the script's own
//! outgoing-input-audit check). See `tools/e2e/record.sh`'s own doc comment for exactly what it
//! does; see `~/aiddnet/data/logs/e2e-8.4a/<timestamp>/` for that run's own logs and recordings
//! afterwards regardless of pass/fail.

use std::env;
use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "touches the real local ddnet-local.service; run explicitly with DDAI_E2E=1"]
fn e2e_record_against_the_local_server() {
    if env::var_os("DDAI_E2E").as_deref() != Some(std::ffi::OsStr::new("1")) {
        eprintln!(
            "skipping: this test touches the real local ddnet-local.service — set DDAI_E2E=1 to \
             actually run tools/e2e/record.sh"
        );
        return;
    }

    // `crates/ddnet-ai` -> `crates` -> the repo root.
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ddnet-ai crate is two levels under the repo root")
        .to_path_buf();
    let script = repo_root.join("tools").join("e2e").join("record.sh");
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
        .expect("failed to spawn tools/e2e/record.sh");
    assert!(
        status.success(),
        "tools/e2e/record.sh exited with {status:?} — see \
         ~/aiddnet/data/logs/e2e-8.4a/<timestamp>/ for that run's own per-check logs"
    );
}
