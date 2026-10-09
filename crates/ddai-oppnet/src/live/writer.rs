//! The live log's file: size-bounded, rotated, written by its own thread so that a slow disk never holds up the bot.
//!
//! [`RotatingFile`] appends whole chunks of JSONL lines to `<path>`; when the next chunk would take the file past `max_bytes` it is renamed
//! to `<path>.1` (the older ones shift to `.2`, ...; at most `keep` files exist, the oldest is deleted) and a new file starts with the
//! header line, so every file says what produced it. [`LogWriter`] feeds one from a bounded channel.

use ddai_os::private::OwnerOnly;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;

/// Default bound of one file (8 MiB) and number of files (4): at most 32 MiB of log, about 1.5 hours of fighting at full rate.
pub const DEFAULT_MAX_BYTES: u64 = 8 << 20;
pub const DEFAULT_KEEP: usize = 4;

pub struct RotatingFile {
    path: PathBuf,
    max_bytes: u64,
    keep: usize,
    header: String,
    file: Option<File>,
    size: u64,
    /// The header of this run is in the file (written at the run's first write, also into a file that already exists, and into every file a rotation starts).
    announced: bool,
}

impl RotatingFile {
    pub fn new(path: PathBuf, max_bytes: u64, keep: usize, header: String) -> RotatingFile {
        RotatingFile {
            path,
            max_bytes: max_bytes.max(1024),
            keep: keep.max(1),
            header,
            file: None,
            size: 0,
            announced: false,
        }
    }

    fn numbered(&self, n: usize) -> PathBuf {
        let mut s = self.path.as_os_str().to_owned();
        s.push(format!(".{n}"));
        PathBuf::from(s)
    }

    fn open(&mut self) -> std::io::Result<()> {
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .owner_only()
            .open(&self.path)?;
        self.size = f.metadata()?.len();
        // Windows: the ACL (Unix: the mode was given at creation); best effort, the folder is the user's own.
        #[cfg(not(unix))]
        if self.size == 0 {
            let _ = ddai_os::private::restrict_file(&self.path);
        }
        if self.size == 0 || !self.announced {
            f.write_all(self.header.as_bytes())?;
            self.size += self.header.len() as u64;
            self.announced = true;
        }
        self.file = Some(f);
        Ok(())
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        self.file = None;
        if self.keep <= 1 {
            std::fs::remove_file(&self.path)?;
            return Ok(());
        }
        let _ = std::fs::remove_file(self.numbered(self.keep - 1));
        for n in (1..self.keep - 1).rev() {
            let from = self.numbered(n);
            if from.exists() {
                std::fs::rename(&from, self.numbered(n + 1))?;
            }
        }
        std::fs::rename(&self.path, self.numbered(1))
    }

    /// Appends `chunk` (whole lines). Rotates first when the file would go past its bound (a file always takes at least one chunk).
    pub fn append(&mut self, chunk: &[u8]) -> std::io::Result<()> {
        if chunk.is_empty() {
            return Ok(());
        }
        if self.file.is_none() {
            self.open()?;
        }
        if self.size + chunk.len() as u64 > self.max_bytes && self.size > self.header.len() as u64 {
            self.rotate()?;
            self.open()?;
        }
        let f = self.file.as_mut().expect("opened above");
        f.write_all(chunk)?;
        self.size += chunk.len() as u64;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The thread behind the log file. Dropping it finishes the queue and joins.
pub struct LogWriter {
    tx: Option<SyncSender<Vec<u8>>>,
    join: Option<JoinHandle<()>>,
    dropped: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
}

impl LogWriter {
    pub fn spawn(path: PathBuf, max_bytes: u64, keep: usize, header: String) -> std::io::Result<LogWriter> {
        let (tx, rx) = sync_channel::<Vec<u8>>(16);
        let errors = Arc::new(AtomicU64::new(0));
        let err = Arc::clone(&errors);
        let mut file = RotatingFile::new(path, max_bytes, keep, header);
        let join = std::thread::Builder::new().name("oppnet-log".into()).spawn(move || {
            while let Ok(chunk) = rx.recv() {
                if file.append(&chunk).is_err() {
                    err.fetch_add(1, Ordering::Relaxed);
                }
            }
        })?;
        Ok(LogWriter {
            tx: Some(tx),
            join: Some(join),
            dropped: Arc::new(AtomicU64::new(0)),
            errors,
        })
    }

    /// Hands a chunk to the thread; `false` (and a count) when the queue is full: the log loses lines, the bot never waits.
    pub fn send(&self, chunk: Vec<u8>) -> bool {
        if chunk.is_empty() {
            return true;
        }
        let Some(tx) = &self.tx else { return false };
        match tx.try_send(chunk) {
            Ok(()) => true,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Chunks dropped because the queue was full, and write errors, since the start.
    pub fn losses(&self) -> (u64, u64) {
        (
            self.dropped.load(Ordering::Relaxed),
            self.errors.load(Ordering::Relaxed),
        )
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn a_file_is_bounded_and_rotated_and_each_file_starts_with_the_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/oppnet-live.jsonl");
        let header = "{\"ev\":\"open\"}\n".to_string();
        let mut f = RotatingFile::new(path.clone(), 1024, 3, header.clone());
        let line = format!("{}\n", "x".repeat(99)); // 100 bytes
        for _ in 0..60 {
            f.append(line.as_bytes()).unwrap();
        }
        // 60 lines of 100 bytes in files of at most 1024: more than 3 files' worth, so the oldest are gone.
        let sizes: Vec<u64> = ["", ".1", ".2"]
            .iter()
            .map(|s| {
                std::fs::metadata(format!("{}{s}", path.display()))
                    .map(|m| m.len())
                    .unwrap_or(0)
            })
            .collect();
        assert!(sizes.iter().all(|&s| s > 0 && s <= 1024), "{sizes:?}");
        assert!(!dir.path().join("sub/oppnet-live.jsonl.3").exists(), "keep = 3 files");
        for s in ["", ".1", ".2"] {
            let l = lines(Path::new(&format!("{}{s}", path.display())));
            assert_eq!(l[0], header.trim_end(), "file {s:?} starts with the header");
        }
        #[cfg(unix)]
        {
            let mode =
                std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&path).unwrap().permissions()) & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn an_existing_file_is_continued_and_every_run_opens_with_its_own_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("l.jsonl");
        {
            let mut f = RotatingFile::new(path.clone(), 4096, 2, "H1\n".into());
            f.append(b"a\n").unwrap();
            f.append(b"a2\n").unwrap();
        }
        // A restart (another model, another server) appends its own header: the lines after it are not the first run's.
        let mut f = RotatingFile::new(path.clone(), 4096, 2, "H2\n".into());
        f.append(b"b\n").unwrap();
        f.append(b"b2\n").unwrap();
        assert_eq!(lines(&path), ["H1", "a", "a2", "H2", "b", "b2"]);
    }

    #[test]
    fn the_writer_thread_flushes_everything_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("l.jsonl");
        {
            let w = LogWriter::spawn(path.clone(), 1 << 20, 2, "H\n".into()).unwrap();
            for i in 0..5 {
                assert!(w.send(format!("line {i}\n").into_bytes()));
            }
            assert!(w.send(Vec::new()), "an empty chunk is a no-op");
            assert_eq!(w.losses(), (0, 0));
        }
        assert_eq!(lines(&path), ["H", "line 0", "line 1", "line 2", "line 3", "line 4"]);
    }

    #[test]
    fn a_blocked_writer_loses_chunks_instead_of_blocking_the_caller() {
        let dir = tempfile::tempdir().unwrap();
        // The path is a directory: every write fails, the thread keeps draining; the sender never blocks either way.
        let w = LogWriter::spawn(dir.path().to_path_buf(), 1 << 20, 2, "H\n".into()).unwrap();
        for _ in 0..200 {
            w.send(b"x\n".to_vec());
        }
        drop(w);
    }
}
