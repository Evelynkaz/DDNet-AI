//! Reading a small file the owner's web unit may have written, without trusting what is at the path (task 5.12, D-099).
//!
//! The root launcher helper, the bot and the web all read files that the web process can create (the favourites, the proxy
//! profiles). A symlink swapped in, a FIFO that would block the reader, or a huge file must never matter: the file is opened
//! with `O_NOFOLLOW | O_NONBLOCK`, its type and size are checked on the opened descriptor, and at most `max` bytes are read.

use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;

/// Why a file could not be read as a small regular file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SafeReadError {
    #[error("the file does not exist")]
    Missing,
    #[error("not a regular file (a symlink, a directory or a special file)")]
    NotRegular,
    #[error("the file is too large")]
    TooLarge,
    #[error("the file could not be read")]
    Io,
}

/// A small regular file's bytes and its modification time (unix seconds, from the opened descriptor).
pub struct SafeRead {
    pub bytes: Vec<u8>,
    pub mtime: u64,
    /// The file's mode bits (`st_mode & 0o7777`).
    pub mode: u32,
    /// The file's owner.
    pub uid: u32,
}

/// Reads a regular file of at most `max` bytes. A symlink is refused (`O_NOFOLLOW`), a FIFO cannot block the reader
/// (`O_NONBLOCK`), and the type is checked on the opened descriptor, so the file checked is the file read.
pub fn read_regular_nofollow(path: &Path, max: usize) -> Result<SafeRead, SafeReadError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => SafeReadError::Missing,
            _ if e.raw_os_error() == Some(libc::ELOOP) => SafeReadError::NotRegular,
            _ => SafeReadError::Io,
        })?;
    let meta = file.metadata().map_err(|_| SafeReadError::Io)?;
    if !meta.is_file() {
        return Err(SafeReadError::NotRegular);
    }
    if meta.len() > max as u64 {
        return Err(SafeReadError::TooLarge);
    }
    let mut bytes = Vec::with_capacity(usize::try_from(meta.len()).unwrap_or(0));
    (&mut file)
        .take(max as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| SafeReadError::Io)?;
    if bytes.len() > max {
        return Err(SafeReadError::TooLarge);
    }
    Ok(SafeRead {
        bytes,
        mtime: u64::try_from(meta.mtime()).unwrap_or(0),
        mode: meta.mode() & 0o7777,
        uid: meta.uid(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_small_regular_files_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a");
        std::fs::write(&file, b"hello").unwrap();
        let r = read_regular_nofollow(&file, 16).unwrap();
        assert_eq!(r.bytes, b"hello");
        assert!(matches!(read_regular_nofollow(&file, 4), Err(SafeReadError::TooLarge)));
        assert!(matches!(
            read_regular_nofollow(&dir.path().join("none"), 16),
            Err(SafeReadError::Missing)
        ));
        let link = dir.path().join("l");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert!(matches!(
            read_regular_nofollow(&link, 16),
            Err(SafeReadError::NotRegular)
        ));
        assert!(matches!(
            read_regular_nofollow(dir.path(), 16),
            Err(SafeReadError::NotRegular)
        ));
    }
}
