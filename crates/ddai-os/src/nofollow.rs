//! Opening a file the caller does not fully trust, without following a symlink.
//!
//! The bot, the web unit and the launcher helper read small files that another process may have written (favourites, proxy
//! profiles, run files). A symlink swapped in, or (Unix) a FIFO that would block the reader, must never matter.

use std::fs::{File, Metadata, OpenOptions};
use std::io;
use std::path::Path;
use std::time::UNIX_EPOCH;

/// Opens `path` for reading without following a symlink at its last component.
///
/// * Unix: `O_NOFOLLOW | O_NONBLOCK`; a symlink makes the open fail with `ELOOP` ([`is_symlink_refusal`]), a FIFO does not block.
/// * Windows: `FILE_FLAG_OPEN_REPARSE_POINT`, so a symlink or junction is opened itself, not its target; the caller's
///   `metadata().is_file()` check then rejects it (a reparse point is not a regular file).
pub fn open_read_nofollow(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    imp::nofollow(&mut options);
    options.open(path)
}

/// Whether `path` exists and is neither a regular file nor a symlink (a directory, a device...), without following a symlink. Windows cannot
/// open a directory with a plain `File::open` (it fails with "access denied"), so when [`open_read_nofollow`] fails the caller asks
/// this to tell "a directory" (a truthful `NotRegular`) from a real I/O error; on Unix the open succeeds and the metadata of the
/// opened file tells, so this is not needed there but gives the same answer.
#[must_use]
pub fn is_existing_non_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| !m.is_file() && !m.file_type().is_symlink())
}

/// Creates `path` for writing, exclusively (it must not exist), never through a symlink at the last component. `unix_mode` is the
/// permission bits it is created with on Unix (cut by the umask: follow up with [`set_unix_mode`]); Windows ignores it.
pub fn create_new_nofollow(path: &Path, unix_mode: u32) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    imp::creation(&mut options, unix_mode);
    options.open(path)
}

/// Opens `path` for writing, creating it if needed and truncating it, never through a symlink at the last component (Unix:
/// `O_NOFOLLOW`; a symlink makes the open fail with `ELOOP`). `unix_mode` as in [`create_new_nofollow`].
pub fn create_truncate_nofollow(path: &Path, unix_mode: u32) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    imp::creation(&mut options, unix_mode);
    options.open(path)
}

/// Sets the Unix permission bits of an open file exactly (the umask cuts the creation mode); a no-op on Windows.
pub fn set_unix_mode(file: &File, unix_mode: u32) -> io::Result<()> {
    imp::set_mode(file, unix_mode)
}

/// Whether `error` is the OS refusing to follow a symlink (Unix `ELOOP`). Always false on Windows, where the open succeeds and the
/// metadata tells.
#[must_use]
pub fn is_symlink_refusal(error: &io::Error) -> bool {
    imp::is_symlink_refusal(error)
}

/// The modification time in seconds since the Unix epoch (0 if unknown or before 1970).
#[must_use]
pub fn mtime_unix(meta: &Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs())
}

/// The Unix mode bits (`st_mode & 0o7777`) and owner of the file; `None` where the OS has neither.
#[must_use]
pub fn unix_mode_and_owner(meta: &Metadata) -> Option<(u32, u32)> {
    imp::mode_and_owner(meta)
}

#[cfg(unix)]
mod imp {
    use std::fs::{Metadata, OpenOptions};
    use std::io;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    pub fn nofollow(options: &mut OpenOptions) {
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }

    pub fn is_symlink_refusal(error: &io::Error) -> bool {
        error.raw_os_error() == Some(libc::ELOOP)
    }

    pub fn mode_and_owner(meta: &Metadata) -> Option<(u32, u32)> {
        Some((meta.mode() & 0o7777, meta.uid()))
    }

    pub fn creation(options: &mut OpenOptions, unix_mode: u32) {
        options.mode(unix_mode).custom_flags(libc::O_NOFOLLOW);
    }

    pub fn set_mode(file: &std::fs::File, unix_mode: u32) -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(unix_mode))
    }
}

#[cfg(not(unix))]
mod imp {
    use std::fs::{Metadata, OpenOptions};
    use std::io;

    #[cfg(windows)]
    pub fn nofollow(options: &mut OpenOptions) {
        use std::os::windows::fs::OpenOptionsExt;
        /// `FILE_FLAG_OPEN_REPARSE_POINT`
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }

    #[cfg(not(windows))]
    pub fn nofollow(_options: &mut OpenOptions) {}

    pub fn is_symlink_refusal(_error: &io::Error) -> bool {
        false
    }

    pub fn mode_and_owner(_meta: &Metadata) -> Option<(u32, u32)> {
        None
    }

    pub fn creation(_options: &mut OpenOptions, _unix_mode: u32) {}

    pub fn set_mode(_file: &std::fs::File, _unix_mode: u32) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn only_a_directory_or_the_like_is_an_existing_non_file() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f");
        std::fs::write(&f, b"x").unwrap();
        assert!(!is_existing_non_file(&f), "a regular file");
        assert!(!is_existing_non_file(&dir.path().join("none")), "a missing path");
        assert!(is_existing_non_file(dir.path()), "a directory");
        #[cfg(unix)]
        {
            let link = dir.path().join("l");
            std::os::unix::fs::symlink(&f, &link).unwrap();
            assert!(!is_existing_non_file(&link), "a symlink is its own case (ELOOP)");
        }
    }

    #[test]
    fn a_regular_file_opens_and_reads() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        std::fs::write(&p, b"abc").unwrap();
        let mut f = open_read_nofollow(&p).unwrap();
        let meta = f.metadata().unwrap();
        assert!(meta.is_file());
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        assert_eq!(s, "abc");
        assert!(mtime_unix(&meta) > 1_500_000_000, "mtime {}", mtime_unix(&meta));
    }

    #[test]
    fn a_missing_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let e = open_read_nofollow(&dir.path().join("nope")).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
        assert!(!is_symlink_refusal(&e));
    }

    #[test]
    fn exclusive_creation_refuses_an_existing_path_and_sets_the_mode() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("new");
        let f = create_new_nofollow(&p, 0o644).unwrap();
        set_unix_mode(&f, 0o644).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o644);
        }
        drop(f);
        assert!(create_new_nofollow(&p, 0o600).is_err(), "exists");
        // Truncating open keeps working on an existing file.
        std::fs::write(&p, b"long old content").unwrap();
        let f = create_truncate_nofollow(&p, 0o644).unwrap();
        drop(f);
        assert_eq!(std::fs::read(&p).unwrap(), b"");
    }

    #[cfg(unix)]
    #[test]
    fn creation_never_follows_a_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        std::fs::write(&victim, b"keep").unwrap();
        let link = dir.path().join("l");
        std::os::unix::fs::symlink(&victim, &link).unwrap();
        assert!(create_new_nofollow(&link, 0o644).is_err());
        let e = create_truncate_nofollow(&link, 0o644).unwrap_err();
        assert!(is_symlink_refusal(&e), "{e:?}");
        assert_eq!(std::fs::read(&victim).unwrap(), b"keep");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("t");
        std::fs::write(&target, b"x").unwrap();
        let link = dir.path().join("l");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let e = open_read_nofollow(&link).unwrap_err();
        assert!(is_symlink_refusal(&e), "{e:?}");
        let (mode, _uid) = unix_mode_and_owner(&std::fs::metadata(&target).unwrap()).unwrap();
        assert_ne!(mode, 0);
    }

    /// A symlink needs a privilege (or developer mode) on Windows: test it when the runner can make one.
    #[cfg(windows)]
    #[test]
    fn a_symlink_is_not_a_regular_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("t");
        std::fs::write(&target, b"x").unwrap();
        let link = dir.path().join("l");
        if std::os::windows::fs::symlink_file(&target, &link).is_err() {
            eprintln!("cannot create a symlink on this runner; skipped");
            return;
        }
        let f = open_read_nofollow(&link).unwrap();
        assert!(
            !f.metadata().unwrap().is_file(),
            "a symlink must not look like a regular file"
        );
    }
}
