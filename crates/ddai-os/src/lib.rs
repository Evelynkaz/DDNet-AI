//! The operating-system seam of DDNet-AI (task 5.5a, D-047/D-127): everything that differs between Linux and Windows lives
//! behind this crate's API, so the rest of the workspace has no `std::os::unix`, `/proc`, `$HOME` or `mode(0o600)` of its own.
//!
//! * [`dirs`]: where the bot's data lives (`~/aiddnet/data` on Linux, `%USERPROFILE%\ddnet-ai\data` on Windows, overridable with
//!   `DDNET_AI_DATA_DIR` and, per command, `--data-dir`), `~` expansion.
//! * [`private`]: owner-only files and directories. Unix permission bits (`0600`/`0700`); on Windows an ACL that grants the current
//!   user alone (set with the system's `icacls`), or, if that fails, a loud warning and the permissions the parent directory gave.
//! * [`nofollow`]: opening a file without following a symlink (and without blocking on a FIFO on Unix).
//! * [`host`]: the load average of the machine, where the OS has one.
//! * [`ipc`]: Unix-domain sockets where they exist; on Windows types nothing can construct (the bot's control channel and live bridge
//!   are then "not available", see that module).
//! * [`marker`]: "is this switch-off marker file there?" with one fail-safe rule on every platform (unknown counts as present).
//! * [`random`]: bytes from the operating system's random generator.
//!
//! No `unsafe`; the only dependencies are `getrandom` and (Unix) `libc` for the open flags. The systemd units, the root launcher and
//! the Unix sockets of the VPS deployment are Linux-only features of the other crates and stay behind `cfg(unix)` there.

pub mod dirs;
pub mod host;
pub mod ipc;
pub mod marker;
pub mod nofollow;
pub mod private;
pub mod random;
