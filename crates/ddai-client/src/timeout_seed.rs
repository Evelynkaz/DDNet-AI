//! The bot's persistent timeout seed (task 4.10, D-100): `cl_timeout_seed` of the official client, kept in a file of the bot's data dir.
//!
//! The seed is what makes the timeout code the same after a restart or a reconnect (the code is derived from it and the server's address,
//! `ddai_net::timeout_code`), and what keeps it unguessable to anyone else, so the file is a secret: it is made with mode `0600` (created
//! exclusively, never through a symlink), read back without following symlinks, a file that is readable by others is tightened to `0600`,
//! and the seed is never logged (its `Debug` shows nothing; the errors here carry no content). It is never in git: the data dir is outside
//! the repository.
//!
//! A seed that cannot be loaded or made is **not fatal**: the bot then plays without the `/timeout` (the caller logs the reason and leaves
//! [`crate::ClientConfig::timeout_seed`] empty), as it did before the timeout code existed.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use ddai_net::timeout_code::{SEED_LEN, TimeoutSeed};

use crate::safe_file::{self, SafeReadError};

/// The seed file inside the bot's directory (`<data-dir>/bot/`).
pub const FILE_NAME: &str = "timeout-seed";

/// Where the bot keeps its seed: `<data-dir>/bot/timeout-seed`.
pub fn path_in(data_dir: &Path) -> PathBuf {
    data_dir.join("bot").join(FILE_NAME)
}

/// Why the seed could not be had. Never carries the seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SeedError {
    #[error("the seed file is not a regular file (a symlink, a directory or a special file)")]
    NotRegular,
    #[error("the seed file does not hold a valid seed (16 characters of DDNet's password alphabet)")]
    Corrupt,
    #[error("the seed file could not be read, made or protected (mode 0600)")]
    Io,
}

/// Reads the seed from `path`, or makes a new random one there (mode `0600`) when the file does not exist. An existing file that others can
/// read is tightened to `0600`.
pub fn load_or_create(path: &Path) -> Result<TimeoutSeed, SeedError> {
    let seed = load_or_create_unprotected(path)?;
    let meta = fs::metadata(path).map_err(|_| SeedError::Io)?;
    if meta.permissions().mode() & 0o077 != 0 {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|_| SeedError::Io)?;
    }
    Ok(seed)
}

fn load_or_create_unprotected(path: &Path) -> Result<TimeoutSeed, SeedError> {
    match read(path) {
        Err(SafeReadError::Missing) => {}
        other => return other.map_err(into_error).and_then(finish),
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|_| SeedError::Io)?;
    }
    let mut random = [0u8; SEED_LEN];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut random))
        .map_err(|_| SeedError::Io)?;
    let seed = TimeoutSeed::from_random(random);
    // `create_new` is `O_EXCL`: it fails on an existing path (a symlink included) instead of following it.
    match OpenOptions::new().write(true).create_new(true).mode(0o600).open(path) {
        Ok(mut file) => {
            file.write_all(format!("{}\n", seed.as_str()).as_bytes())
                .and_then(|()| file.sync_all())
                .map_err(|_| SeedError::Io)?;
            Ok(seed)
        }
        // Another process made it first: use theirs.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => read(path).map_err(into_error).and_then(finish),
        Err(_) => Err(SeedError::Io),
    }
}

fn read(path: &Path) -> Result<safe_file::SafeRead, SafeReadError> {
    safe_file::read_regular_nofollow(path, 256)
}

fn into_error(e: SafeReadError) -> SeedError {
    match e {
        SafeReadError::NotRegular => SeedError::NotRegular,
        SafeReadError::Missing | SafeReadError::TooLarge | SafeReadError::Io => SeedError::Io,
    }
}

/// Checks the bytes of an existing file.
fn finish(file: safe_file::SafeRead) -> Result<TimeoutSeed, SeedError> {
    std::str::from_utf8(&file.bytes)
        .ok()
        .and_then(TimeoutSeed::parse)
        .ok_or(SeedError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn mode(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o7777
    }

    #[test]
    fn the_first_call_makes_a_0600_file_and_later_calls_read_the_same_seed() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(dir.path());
        assert!(!path.exists());
        let a = load_or_create(&path).unwrap();
        assert_eq!(mode(&path), 0o600);
        assert_eq!(
            fs::read_to_string(&path).unwrap().len(),
            SEED_LEN + 1,
            "16 characters and a newline"
        );
        let b = load_or_create(&path).unwrap();
        assert_eq!(a, b, "the same seed after a restart");
    }

    #[test]
    fn two_files_get_two_different_seeds() {
        let dir = tempfile::tempdir().unwrap();
        let a = load_or_create(&dir.path().join("a")).unwrap();
        let b = load_or_create(&dir.path().join("b")).unwrap();
        assert_ne!(a, b, "the seed comes from OS randomness");
    }

    #[test]
    fn a_file_others_can_read_is_tightened() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seed");
        fs::write(&path, "ABCDEFGHKLMNPRST\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let seed = load_or_create(&path).unwrap();
        assert_eq!(seed.as_str(), "ABCDEFGHKLMNPRST");
        assert_eq!(mode(&path), 0o600);
    }

    #[test]
    fn a_corrupt_file_is_an_error_and_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        for bad in [
            "",
            "short",
            "ABCDEFGHKLMNPRS0\n",
            "ABCDEFGHKLMNPRSTU\n",
            "\u{e9}BCDEFGHKLMNPRST",
        ] {
            let path = dir.path().join("seed");
            fs::write(&path, bad).unwrap();
            assert_eq!(load_or_create(&path).unwrap_err(), SeedError::Corrupt, "{bad:?}");
            assert_eq!(fs::read_to_string(&path).unwrap(), bad, "not overwritten");
            fs::remove_file(&path).unwrap();
        }
        let big = dir.path().join("big");
        fs::write(&big, "A".repeat(1000)).unwrap();
        assert_eq!(load_or_create(&big).unwrap_err(), SeedError::Io);
    }

    #[test]
    fn a_symlink_is_never_followed_for_reading_or_creating() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        fs::write(&target, "ABCDEFGHKLMNPRST\n").unwrap();
        let link = dir.path().join("seed");
        symlink(&target, &link).unwrap();
        assert_eq!(load_or_create(&link).unwrap_err(), SeedError::NotRegular);
        // a dangling link is not "missing": nothing is created through it
        let dangling = dir.path().join("dangling");
        symlink(dir.path().join("nowhere"), &dangling).unwrap();
        assert!(load_or_create(&dangling).is_err());
        assert!(!dir.path().join("nowhere").exists());
        assert_eq!(
            load_or_create(dir.path()).unwrap_err(),
            SeedError::NotRegular,
            "a directory"
        );
    }

    #[test]
    fn the_errors_and_the_seed_never_show_the_seed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seed");
        let seed = load_or_create(&path).unwrap();
        let text = seed.as_str().to_string();
        assert!(!format!("{seed:?}").contains(&text));
        for e in [SeedError::Corrupt, SeedError::Io, SeedError::NotRegular] {
            assert!(!format!("{e} {e:?}").contains(&text));
        }
    }
}
