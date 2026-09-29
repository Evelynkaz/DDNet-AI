//! Per-(server, identity) single-instance guard (review round 1, finding F8; `CLAUDE.md`/D-016:
//! "не больше одного бота на сервере") — shared by `ddnet-ai play` and `ddnet-ai record`, guarding
//! against the accidental-duplicate-launch scenario the finding actually names ("record+play could
//! run against same server simultaneously", i.e. the same command started twice, or a systemd
//! auto-restart racing a still-shutting-down previous instance).
//!
//! **Keyed by `(address, name)`, not address alone.** An earlier version of this fix locked on the
//! address only — which passed every unit test but broke this project's own established local e2e
//! pattern the very first time it ran live: `tools/e2e/session.sh`/`record.sh` (task 2.3/8.4a, both
//! predating this fix) deliberately run *several* distinctly-named bots against the same
//! `127.0.0.1:8303` at once (two `--brain circle` plus a validation bot plus the recorder) to
//! exercise multi-player scenarios — an address-only lock refused every one of them but the first
//! to connect. D-016's own place in `CLAUDE.md` (the "Живая игра" section, about real servers, not
//! local test infrastructure) and its literal wording ("не больше одного бота на сервере") are
//! about not colliding with — or looking like spam on — a real community server under one
//! identity; keying by `(address, name)` still refuses the actual failure mode named in the finding
//! (the same identity connecting twice) while leaving this project's own local multi-bot testing,
//! and any future legitimate multi-account local scenario, unaffected.
//!
//! An advisory, OS-level file-descriptor lock (`fd-lock`, pure safe Rust, no `unsafe` needed from
//! this crate) on `~/aiddnet/data/run/<sanitized addr>-<sanitized name>.lock` — held for the whole
//! process lifetime, dropped (and thus released) automatically on any exit path, **including a
//! `kill -9`** (the OS itself releases an `flock`-style lock when the holding process dies, no
//! stale-lock cleanup logic needed, unlike a PID file that could be left behind pointing at a dead
//! process).

use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;

/// `~/aiddnet/data/run`, falling back to a relative `data/run` if `$HOME` isn't set — same
/// fallback pattern as every other `default_*` helper in this crate.
pub fn default_run_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) if !home.is_empty() => PathBuf::from(home).join("aiddnet").join("data").join("run"),
        _ => PathBuf::from("data").join("run"),
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum LockError {
    #[error("I/O error preparing the lock file at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error(
        "{addr} already has another ddai-client process connected as {name:?} (lock held at {path}) — \
         CLAUDE.md/D-016: never more than one bot under the same identity on a server"
    )]
    AlreadyHeld {
        addr: SocketAddr,
        name: String,
        path: PathBuf,
    },
}

/// Holds the lock for as long as it lives — drop it (or let it fall out of scope, including on
/// panic/process exit) to release. Deliberately leaks a small, fixed, one-per-process allocation
/// (`Box::leak`, no `unsafe`) to give the returned guard a `'static` lifetime without a
/// self-referential struct: this lock is meant to live for the process's entire run anyway, so
/// "leaking" it is exactly the intended lifetime, not a real leak.
///
/// Review round 3, finding F22: on a clean drop, also removes the lock file itself from disk (not
/// just releasing the `flock` — the two are different: the *directory entry* at `path` otherwise
/// stays behind forever, one per distinct `(address, name)` this project has ever run, which
/// `session.sh`'s randomized bot names in particular can pile up quickly). Best-effort — a
/// `kill -9` still leaves the file behind exactly like before (the OS has no hook to run our own
/// code on an unclean kill; the `flock` itself is still released correctly by the kernel either
/// way, which is what actually matters for correctness) — so this is pure housekeeping on the
/// paths that already run some cleanup, never something a re-acquire anywhere depends on.
pub struct ServerLock {
    _guard: fd_lock::RwLockWriteGuard<'static, std::fs::File>,
    path: PathBuf,
}

impl Drop for ServerLock {
    fn drop(&mut self) {
        // Best-effort: a missing file (already removed, e.g. by a concurrent cleanup) or a
        // permissions error must never turn a normal shutdown into a panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Acquires the lock for `(addr, name)`, or returns [`LockError::AlreadyHeld`] if another process
/// (this one or a sibling `ddnet-ai play`/`record`) already holds it *under the same name on the
/// same address* — see the module's own doc comment for why the key includes `name`. Callers
/// should treat any error here as fatal — refuse to start, per D-016.
pub fn acquire(addr: SocketAddr, name: &str) -> Result<ServerLock, LockError> {
    let dir = default_run_dir();
    std::fs::create_dir_all(&dir).map_err(|source| LockError::Io {
        path: dir.clone(),
        source,
    })?;
    let path = dir.join(format!("{}-{}.lock", sanitize(&addr.to_string()), sanitize(name)));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|source| LockError::Io {
            path: path.clone(),
            source,
        })?;

    // Leaked deliberately — see `ServerLock`'s own doc comment.
    let lock: &'static mut fd_lock::RwLock<std::fs::File> = Box::leak(Box::new(fd_lock::RwLock::new(file)));
    let guard = lock.try_write().map_err(|_| LockError::AlreadyHeld {
        addr,
        name: name.to_string(),
        path: path.clone(),
    })?;
    Ok(ServerLock { _guard: guard, path })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_replaces_unsafe_characters() {
        assert_eq!(sanitize("127.0.0.1:8303"), "127_0_0_1_8303");
    }

    /// Isolated run-dir (own temp dir, never `default_run_dir()`, which reads the process-wide
    /// `$HOME` and is not test-seamed) so this can never collide with a real
    /// `~/aiddnet/data/run` lock file from an actual live process, or with other tests running in
    /// parallel.
    fn acquire_in(dir: &std::path::Path, addr: SocketAddr, name: &str) -> Result<ServerLock, LockError> {
        let path = dir.join(format!("{}-{}.lock", sanitize(&addr.to_string()), sanitize(name)));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|source| LockError::Io {
                path: path.clone(),
                source,
            })?;
        let lock: &'static mut fd_lock::RwLock<std::fs::File> = Box::leak(Box::new(fd_lock::RwLock::new(file)));
        let guard = lock.try_write().map_err(|_| LockError::AlreadyHeld {
            addr,
            name: name.to_string(),
            path: path.clone(),
        })?;
        Ok(ServerLock { _guard: guard, path })
    }

    #[test]
    fn second_acquire_for_the_same_address_and_name_is_refused_while_the_first_is_held() {
        let dir = tempfile::tempdir().unwrap();
        let addr: SocketAddr = "127.0.0.1:19999".parse().unwrap();
        let _guard1 = acquire_in(dir.path(), addr, "SameName").expect("first acquire must succeed");
        assert!(
            acquire_in(dir.path(), addr, "SameName").is_err(),
            "second acquire under the same (address, name) must be refused"
        );
    }

    /// Review round 1, finding F8's fix, corrected: an address-only lock broke this project's own
    /// `tools/e2e/session.sh`/`record.sh`, which deliberately run several distinctly-named bots
    /// against the same local server at once (see the module's own doc comment) — this is the
    /// regression test for that, proving the fix (keying by `(address, name)`) actually allows it.
    #[test]
    fn different_names_on_the_same_address_do_not_collide() {
        let dir = tempfile::tempdir().unwrap();
        let addr: SocketAddr = "127.0.0.1:19998".parse().unwrap();
        let _guard_a = acquire_in(dir.path(), addr, "e2aC1").expect("first identity must succeed");
        let _guard_b =
            acquire_in(dir.path(), addr, "e2aC2").expect("a different identity on the same address must succeed too");
        let _guard_c = acquire_in(dir.path(), addr, "RecE2E").expect("and a third");
    }

    #[test]
    fn acquire_succeeds_and_releases_on_drop_allowing_a_later_acquire() {
        let dir = tempfile::tempdir().unwrap();
        let addr: SocketAddr = "127.0.0.1:18888".parse().unwrap();

        {
            let _guard = acquire_in(dir.path(), addr, "SomeName").expect("first acquire must succeed");
            // guard (and the lock) drop at the end of this block.
        }

        assert!(
            acquire_in(dir.path(), addr, "SomeName").is_ok(),
            "a released lock must be re-acquirable"
        );
    }

    /// Review round 3, finding F22: a clean drop must also remove the lock file itself from disk,
    /// not just release the `flock` — otherwise one file per distinct `(address, name)` this
    /// project has ever run piles up forever under `~/aiddnet/data/run/`.
    #[test]
    fn dropping_the_lock_cleanly_removes_the_file_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let addr: SocketAddr = "127.0.0.1:18887".parse().unwrap();
        let path = dir.path().join(format!(
            "{}-{}.lock",
            sanitize(&addr.to_string()),
            sanitize("CleanExit")
        ));

        {
            let _guard = acquire_in(dir.path(), addr, "CleanExit").expect("first acquire must succeed");
            assert!(path.exists(), "the lock file must exist while held");
        }

        assert!(!path.exists(), "the lock file must be gone after a clean drop");
    }
}
