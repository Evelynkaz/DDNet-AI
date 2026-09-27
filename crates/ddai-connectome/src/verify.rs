//! Checking a local file against its pinned [`FileEntry`] — the offline-testable core of
//! `fetch`'s "refuses on any mismatch" contract. No network access here: given a path already on
//! disk, decide whether it matches what the manifest pins.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::hashing::{FileHashes, hash_file};
use crate::manifest::FileEntry;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("{path}: cannot read file: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: size mismatch: manifest says {expected} bytes, file is {actual} bytes")]
    SizeMismatch { path: PathBuf, expected: u64, actual: u64 },
    #[error("{path}: md5 mismatch: manifest says {expected}, computed {actual}")]
    Md5Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("{path}: sha256 mismatch: manifest says {expected}, computed {actual}")]
    Sha256Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
}

/// Verifies that the file at `path` matches `entry`'s pinned `size` and `md5` exactly, and its
/// `sha256` too if one is pinned (an empty `sha256` means "not pinned yet" and is skipped, not
/// treated as a match). Returns the computed hashes on success so callers (like `fetch
/// --update-sha256`) can reuse them without hashing the file twice.
pub fn verify_local_file(path: &Path, entry: &FileEntry) -> Result<FileHashes, VerifyError> {
    let hashes = hash_file(path).map_err(|source| VerifyError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    if hashes.size != entry.size {
        return Err(VerifyError::SizeMismatch {
            path: path.to_path_buf(),
            expected: entry.size,
            actual: hashes.size,
        });
    }
    if !hashes.md5_hex.eq_ignore_ascii_case(&entry.md5) {
        return Err(VerifyError::Md5Mismatch {
            path: path.to_path_buf(),
            expected: entry.md5.clone(),
            actual: hashes.md5_hex.clone(),
        });
    }
    if entry.has_sha256() && !hashes.sha256_hex.eq_ignore_ascii_case(&entry.sha256) {
        return Err(VerifyError::Sha256Mismatch {
            path: path.to_path_buf(),
            expected: entry.sha256.clone(),
            actual: hashes.sha256_hex.clone(),
        });
    }
    Ok(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_for(content: &[u8], sha256: &str) -> FileEntry {
        let hashes = hash_file_from_bytes(content);
        FileEntry {
            name: "f.feather".into(),
            url: "https://example.com/f.feather".into(),
            generation: 1,
            size: hashes.size,
            md5: hashes.md5_hex,
            sha256: sha256.to_string(),
            license: "CC-BY-4.0".into(),
            source_version: "v1".into(),
            citation: "x".into(),
        }
    }

    fn hash_file_from_bytes(content: &[u8]) -> FileHashes {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tmp.bin");
        std::fs::write(&path, content).unwrap();
        hash_file(&path).unwrap()
    }

    fn write_temp(content: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.feather");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn matching_file_verifies_ok_including_sha256() {
        let content = b"hello connectome";
        let hashes = hash_file_from_bytes(content);
        let entry = entry_for(content, &hashes.sha256_hex);
        let (_dir, path) = write_temp(content);
        let result = verify_local_file(&path, &entry).unwrap();
        assert_eq!(result.sha256_hex, hashes.sha256_hex);
    }

    #[test]
    fn matching_file_verifies_ok_when_sha256_not_pinned_yet() {
        let content = b"hello connectome";
        let entry = entry_for(content, "");
        let (_dir, path) = write_temp(content);
        assert!(verify_local_file(&path, &entry).is_ok());
    }

    #[test]
    fn detects_tampered_size() {
        let content = b"original bytes";
        let entry = entry_for(content, "");
        let (_dir, path) = write_temp(b"original bytes, but longer now");
        let err = verify_local_file(&path, &entry).unwrap_err();
        assert!(
            matches!(err, VerifyError::SizeMismatch { .. }),
            "expected SizeMismatch, got {err:?}"
        );
    }

    #[test]
    fn detects_tampered_content_same_size_wrong_md5() {
        let original = b"AAAAAAAAAA";
        let tampered = b"BBBBBBBBBB"; // same length, different bytes -> md5 differs, size matches
        assert_eq!(original.len(), tampered.len());
        let entry = entry_for(original, "");
        let (_dir, path) = write_temp(tampered);
        let err = verify_local_file(&path, &entry).unwrap_err();
        assert!(
            matches!(err, VerifyError::Md5Mismatch { .. }),
            "expected Md5Mismatch, got {err:?}"
        );
    }

    #[test]
    fn detects_tampered_sha256_with_correct_size_and_md5() {
        // Craft an entry whose md5/size match the tampered file (as if only the manifest's
        // sha256 pin were stale or the file's content changed in a way that happens to collide
        // on size, which we simulate directly here) so the sha256 check is the one that fires.
        let content = b"same size and md5, wrong sha256 pin";
        let (_dir, path) = write_temp(content);
        let real = hash_file(&path).unwrap();
        let mut entry = entry_for(content, "");
        entry.sha256 = "0".repeat(64); // clearly wrong, but well-formed hex-length string
        assert_ne!(entry.sha256, real.sha256_hex);
        let err = verify_local_file(&path, &entry).unwrap_err();
        assert!(
            matches!(err, VerifyError::Sha256Mismatch { .. }),
            "expected Sha256Mismatch, got {err:?}"
        );
    }

    #[test]
    fn missing_file_reports_io_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.feather");
        let entry = entry_for(b"whatever", "");
        let err = verify_local_file(&path, &entry).unwrap_err();
        assert!(matches!(err, VerifyError::Io { .. }), "expected Io, got {err:?}");
    }

    #[test]
    fn md5_comparison_is_case_insensitive() {
        let content = b"case insensitivity check";
        let hashes = hash_file_from_bytes(content);
        let mut entry = entry_for(content, "");
        entry.md5 = hashes.md5_hex.to_uppercase();
        let (_dir, path) = write_temp(content);
        assert!(verify_local_file(&path, &entry).is_ok());
    }
}
