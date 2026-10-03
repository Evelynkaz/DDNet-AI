//! Path validation for the training panel: every file the web reads under the runs root goes through here.
//!
//! The rules (task 5.8):
//!
//! * a name that comes from a request (experiment, run) is an **identifier**: 1..=[`MAX_IDENT_LEN`] bytes of
//!   `[A-Za-z0-9._-]`, never starting with a dot. No separator, no `..`, no NUL, no space can pass, so a request can never
//!   spell a path outside the directory it is joined to;
//! * the joined path is then **canonicalised** (symlinks resolved) and must still lie under the canonical runs root, so a
//!   symlink inside the runs tree that points outside it is refused, while one that stays inside is fine;
//! * a file must be a regular file (never a FIFO or a device: opening one could block the read forever).
//!
//! What remains is the usual check-then-open gap: the path could be swapped for a symlink between the check and the open.
//! The runs tree is written only by the owner's own training jobs, so this is accepted (and documented in `docs/formats.md`).

use std::path::{Path, PathBuf};

/// The longest identifier (a run or experiment directory name) the panel accepts.
pub const MAX_IDENT_LEN: usize = 96;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PathError {
    /// The name is not a valid identifier (traversal attempts end here).
    #[error("invalid name")]
    Invalid,
    #[error("not found")]
    NotFound,
    /// The resolved path is outside the runs root (a symlink pointing out).
    #[error("outside the runs root")]
    Escapes,
    #[error("unexpected file type")]
    WrongType,
    #[error("i/o error")]
    Io,
}

/// Whether `name` is an acceptable experiment / run / file name: `[A-Za-z0-9._-]{1,96}`, not starting with a dot.
pub fn is_valid_ident(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_IDENT_LEN
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

/// The runs root, canonicalised once per request.
#[derive(Debug, Clone)]
pub struct RunsRoot {
    root: PathBuf,
}

impl RunsRoot {
    /// Opens `path` as the runs root: it must exist and be a directory (a symlink to one is fine; the root is the target).
    pub fn open(path: &Path) -> Result<Self, PathError> {
        let root = std::fs::canonicalize(path).map_err(map_io)?;
        if !root.is_dir() {
            return Err(PathError::WrongType);
        }
        Ok(Self { root })
    }

    pub fn path(&self) -> &Path {
        &self.root
    }

    /// The directory `root/<parts[0]>/<parts[1]>/…`, every part an identifier, resolved and inside the root.
    pub fn dir(&self, parts: &[&str]) -> Result<PathBuf, PathError> {
        self.join(&self.root, parts, Kind::Dir)
    }

    /// The regular file `dir/<name>` (`dir` already validated by [`RunsRoot::dir`]), resolved and inside the root.
    pub fn file(&self, dir: &Path, name: &str) -> Result<PathBuf, PathError> {
        self.join(dir, &[name], Kind::File)
    }

    fn join(&self, base: &Path, parts: &[&str], kind: Kind) -> Result<PathBuf, PathError> {
        if parts.is_empty() || !parts.iter().all(|p| is_valid_ident(p)) {
            return Err(PathError::Invalid);
        }
        let mut path = base.to_path_buf();
        for part in parts {
            path.push(part);
        }
        let resolved = std::fs::canonicalize(&path).map_err(map_io)?;
        if !resolved.starts_with(&self.root) {
            return Err(PathError::Escapes);
        }
        let ok = match kind {
            Kind::Dir => resolved.is_dir(),
            Kind::File => resolved.is_file(),
        };
        if ok { Ok(resolved) } else { Err(PathError::WrongType) }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Dir,
    File,
}

fn map_io(e: std::io::Error) -> PathError {
    match e.kind() {
        std::io::ErrorKind::NotFound => PathError::NotFound,
        _ => PathError::Io,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_accept_run_names_and_refuse_everything_else() {
        for ok in ["E-008", "e008-p1-fly-drop-s1", "7.2b", "a_b.c", "x"] {
            assert!(is_valid_ident(ok), "{ok}");
        }
        for bad in [
            "",
            ".",
            "..",
            ".hidden",
            "a/b",
            "a\\b",
            "../x",
            "a b",
            "a\0b",
            "é",
            "a:b",
            "a%2fb",
            &"x".repeat(MAX_IDENT_LEN + 1),
        ] {
            assert!(!is_valid_ident(bad), "{bad:?}");
        }
        assert!(is_valid_ident(&"x".repeat(MAX_IDENT_LEN)));
    }

    #[test]
    fn dir_and_file_resolve_inside_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("E-1").join("run-a")).unwrap();
        std::fs::write(tmp.path().join("E-1").join("run-a").join("status.json"), "{}").unwrap();
        let root = RunsRoot::open(tmp.path()).unwrap();
        let dir = root.dir(&["E-1", "run-a"]).unwrap();
        assert!(dir.starts_with(root.path()));
        let file = root.file(&dir, "status.json").unwrap();
        assert!(file.is_file());
        assert_eq!(root.file(&dir, "missing.json"), Err(PathError::NotFound));
        assert_eq!(root.dir(&["E-1", "nope"]), Err(PathError::NotFound));
        // A file is not a directory and the other way round.
        assert_eq!(root.dir(&["E-1", "run-a", "status.json"]), Err(PathError::WrongType));
        assert_eq!(
            root.file(&root.dir(&["E-1"]).unwrap(), "run-a"),
            Err(PathError::WrongType)
        );
    }

    #[test]
    fn traversal_attempts_are_invalid_before_any_filesystem_access() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("E-1").join("run-a")).unwrap();
        let root = RunsRoot::open(tmp.path()).unwrap();
        for attempt in [
            vec!["..", "etc"],
            vec!["E-1", ".."],
            vec!["E-1/../E-1"],
            vec!["/etc"],
            vec!["E-1", "run-a/../../.."],
            vec![""],
            vec![],
        ] {
            assert_eq!(root.dir(&attempt), Err(PathError::Invalid), "{attempt:?}");
        }
        let dir = root.dir(&["E-1", "run-a"]).unwrap();
        for name in ["../status.json", "/etc/passwd", "..", "a/b", ""] {
            assert_eq!(root.file(&dir, name), Err(PathError::Invalid), "{name:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_pointing_out_of_the_root_are_refused_but_inside_ones_work() {
        use std::os::unix::fs::symlink;
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.json"), "{\"secret\":1}").unwrap();
        std::fs::create_dir_all(outside.path().join("dir")).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let run = tmp.path().join("E-1").join("run-a");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("real.json"), "{}").unwrap();
        symlink(outside.path().join("secret.json"), run.join("status.json")).unwrap();
        symlink(outside.path().join("dir"), tmp.path().join("E-2")).unwrap();
        symlink(run.join("real.json"), run.join("alias.json")).unwrap();

        let root = RunsRoot::open(tmp.path()).unwrap();
        let dir = root.dir(&["E-1", "run-a"]).unwrap();
        assert_eq!(root.file(&dir, "status.json"), Err(PathError::Escapes));
        assert_eq!(root.dir(&["E-2"]), Err(PathError::Escapes));
        assert!(
            root.file(&dir, "alias.json").is_ok(),
            "a symlink that stays inside is fine"
        );
    }

    #[test]
    fn a_missing_or_file_root_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            RunsRoot::open(&tmp.path().join("absent")).unwrap_err(),
            PathError::NotFound
        );
        std::fs::write(tmp.path().join("f"), "x").unwrap();
        assert_eq!(RunsRoot::open(&tmp.path().join("f")).unwrap_err(), PathError::WrongType);
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_is_not_a_regular_file() {
        let tmp = tempfile::tempdir().unwrap();
        let run = tmp.path().join("E-1").join("run-a");
        std::fs::create_dir_all(&run).unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(run.join("metrics.jsonl"))
            .status();
        if status.is_ok_and(|s| s.success()) {
            let root = RunsRoot::open(tmp.path()).unwrap();
            let dir = root.dir(&["E-1", "run-a"]).unwrap();
            assert_eq!(root.file(&dir, "metrics.jsonl"), Err(PathError::WrongType));
        }
    }
}
