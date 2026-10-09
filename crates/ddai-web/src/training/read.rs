//! Bounded reads of files under the runs root (the path was already validated by [`super::paths`]).
//!
//! Nothing here reads more than a caller-given cap into memory, however large the file is or grows while it is read
//! (`metrics.jsonl` is appended to by a running job): small files are read whole or refused, the big append-only one is read
//! from its tail.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ReadError {
    /// The file is larger than the cap (a status, a config or a summary has no business being that big).
    #[error("file too large")]
    TooLarge,
    #[error("not a regular file")]
    NotFile,
    #[error("i/o error")]
    Io,
}

/// The last bytes of a file, cut at a line boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tail {
    pub bytes: Vec<u8>,
    /// The file was longer than the cap: the first (partial) line is dropped and earlier lines are missing.
    pub truncated: bool,
    /// The length of the whole file when it was opened.
    pub file_len: u64,
}

/// Opens `path` for reading without following a symlink in the **last** component (`O_NOFOLLOW`: a path that was a regular
/// file when it was checked and became a symlink since is refused, not followed) and without blocking (`O_NONBLOCK`: a FIFO
/// swapped in at that name opens at once, and the `fstat` below refuses it, instead of hanging the read forever). The path
/// is already canonical (`paths::RunsRoot`), so its last component is never legitimately a symlink. A directory component
/// swapped for a symlink between the check and the open is not caught by this; see `docs/formats.md` §29.2.
fn open_regular(path: &Path) -> Result<(File, u64), ReadError> {
    let file = ddai_os::nofollow::open_read_nofollow(path).map_err(|_| ReadError::Io)?;
    let meta = file.metadata().map_err(|_| ReadError::Io)?;
    if !meta.is_file() {
        return Err(ReadError::NotFile);
    }
    Ok((file, meta.len()))
}

/// The whole file, or [`ReadError::TooLarge`] when it is longer than `cap` bytes.
pub fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, ReadError> {
    let (file, len) = open_regular(path)?;
    if len > cap {
        return Err(ReadError::TooLarge);
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    // `take(cap + 1)`: a file that grows past the cap while being read is still refused, never buffered.
    file.take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| ReadError::Io)?;
    if bytes.len() as u64 > cap {
        return Err(ReadError::TooLarge);
    }
    Ok(bytes)
}

/// The sha256 of a regular file (lower-case hex), streamed, or [`ReadError::TooLarge`] past `cap` bytes.
pub fn hash_file(path: &Path, cap: u64) -> Result<String, ReadError> {
    use sha2::{Digest, Sha256};
    let (mut file, len) = open_regular(path)?;
    if len > cap {
        return Err(ReadError::TooLarge);
    }
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf).map_err(|_| ReadError::Io)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > cap {
            return Err(ReadError::TooLarge);
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The file's last `cap` bytes, starting at a line boundary: a file at most `cap` long is returned whole.
pub fn read_tail(path: &Path, cap: u64) -> Result<Tail, ReadError> {
    let (mut file, len) = open_regular(path)?;
    let truncated = len > cap;
    if truncated {
        file.seek(SeekFrom::Start(len - cap)).map_err(|_| ReadError::Io)?;
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len.min(cap)).unwrap_or(0));
    file.take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| ReadError::Io)?;
    if truncated {
        // Cut into the middle of a line: drop up to and including the first newline. No newline at all means the cap
        // holds a fragment of a single (huge) line: nothing usable.
        match bytes.iter().position(|&b| b == b'\n') {
            Some(i) => {
                bytes.drain(..=i);
            }
            None => bytes.clear(),
        }
    }
    Ok(Tail {
        bytes,
        truncated,
        file_len: len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file_with(content: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        std::fs::File::create(&path).unwrap().write_all(content).unwrap();
        (dir, path)
    }

    #[test]
    fn capped_read_returns_small_files_and_refuses_big_ones() {
        let (_d, path) = file_with(b"hello");
        assert_eq!(read_capped(&path, 5).unwrap(), b"hello");
        assert_eq!(read_capped(&path, 100).unwrap(), b"hello");
        assert_eq!(read_capped(&path, 4), Err(ReadError::TooLarge));
        assert_eq!(read_capped(&path, 0), Err(ReadError::TooLarge));
    }

    #[test]
    fn capped_read_of_a_missing_file_or_directory_fails() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_capped(&dir.path().join("none"), 10), Err(ReadError::Io));
        assert_eq!(read_capped(dir.path(), 10), Err(ReadError::NotFile));
    }

    /// Runs `f` on a thread and fails the test instead of hanging when it does not return within 3 s.
    #[cfg(unix)]
    fn within_3s<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(std::time::Duration::from_secs(3))
            .expect("the read hung (a FIFO was opened blocking?)")
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_at_the_path_is_refused_without_hanging() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("metrics.jsonl");
        if !std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .is_ok_and(|s| s.success())
        {
            return; // no mkfifo here
        }
        let (a, b, c, d) = (fifo.clone(), fifo.clone(), fifo.clone(), fifo);
        assert_eq!(within_3s(move || read_capped(&a, 100)), Err(ReadError::NotFile));
        assert_eq!(within_3s(move || read_tail(&b, 100)).unwrap_err(), ReadError::NotFile);
        assert_eq!(within_3s(move || hash_file(&c, 100)), Err(ReadError::NotFile));
        assert_eq!(within_3s(move || open_regular(&d).map(|_| ())), Err(ReadError::NotFile));
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_swapped_in_at_the_last_component_is_not_followed() {
        // `RunsRoot` hands out canonical paths, so a symlink here can only be a swap made after the check.
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.json");
        std::fs::write(&target, b"outside").unwrap();
        let link = dir.path().join("status.json");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert_eq!(read_capped(&link, 100), Err(ReadError::Io));
        assert_eq!(read_tail(&link, 100).unwrap_err(), ReadError::Io);
        assert_eq!(hash_file(&link, 100), Err(ReadError::Io));
        assert_eq!(read_capped(&target, 100).unwrap(), b"outside");
    }

    #[test]
    fn hash_matches_the_known_sha256_and_respects_the_cap() {
        let (_d, path) = file_with(b"abc");
        assert_eq!(
            hash_file(&path, 10).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(hash_file(&path, 2), Err(ReadError::TooLarge));
    }

    #[test]
    fn tail_of_a_small_file_is_the_whole_file() {
        let (_d, path) = file_with(b"a\nb\nc\n");
        let tail = read_tail(&path, 6).unwrap();
        assert_eq!(tail.bytes, b"a\nb\nc\n");
        assert!(!tail.truncated);
        assert_eq!(tail.file_len, 6);
        let empty = file_with(b"");
        let tail = read_tail(&empty.1, 10).unwrap();
        assert!(tail.bytes.is_empty() && !tail.truncated);
    }

    #[test]
    fn tail_of_a_big_file_starts_on_a_line_boundary() {
        // 10 lines of "line-N\n" (7 bytes each = 70 bytes); a cap of 20 bytes lands in the middle of a line.
        let content: String = (0..10).map(|i| format!("line-{i}\n")).collect();
        let (_d, path) = file_with(content.as_bytes());
        let tail = read_tail(&path, 20).unwrap();
        assert!(tail.truncated);
        assert_eq!(tail.file_len, 70);
        let text = String::from_utf8(tail.bytes).unwrap();
        assert_eq!(text, "line-8\nline-9\n");
        // A cut that lands exactly on a line start still drops that one (the byte before it is unknown): safe, not wrong.
        let tail = read_tail(&path, 14).unwrap();
        assert_eq!(String::from_utf8(tail.bytes).unwrap(), "line-9\n");
    }

    #[test]
    fn tail_never_returns_more_than_the_cap() {
        let content = "x".repeat(10_000) + "\n" + &"y".repeat(10);
        let (_d, path) = file_with(content.as_bytes());
        let tail = read_tail(&path, 100).unwrap();
        assert!(tail.bytes.len() <= 100);
        // The cap holds only the end of one very long line: it is dropped, not returned half-cut.
        let tail = read_tail(&path, 5).unwrap();
        assert!(tail.bytes.is_empty() && tail.truncated);
    }
}
