//! `fetch --manifest <path> --dest <dir> [--update-sha256]`: downloads, resumes and verifies the
//! files listed in `manifest`. The only module in this crate that touches the network.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::RANGE;

use crate::goog_hash::parse_goog_md5_hex;
use crate::hashing::FileHashes;
use crate::manifest::{FileEntry, Manifest, update_sha256_in_file};
use crate::verify::verify_local_file;

pub struct FetchOptions {
    pub manifest_path: PathBuf,
    pub dest: PathBuf,
    pub update_sha256: bool,
}

#[derive(Debug)]
pub enum FetchAction {
    /// The final file was already present and passed verification; nothing was downloaded.
    AlreadyPresent,
    /// Ends with `entry.size` verified bytes in the final file. `performed_get` is `false` when
    /// this run found an already-complete `.part` from a previous run and just verified +
    /// renamed it (no network access this run); `resumed_from` is 0 for a fresh download.
    Downloaded {
        resumed_from: u64,
        performed_get: bool,
        hashes: FileHashes,
    },
}

#[derive(Debug)]
pub struct FetchOutcome {
    pub name: String,
    pub action: FetchAction,
    /// Set if this call filled in a previously-empty `sha256` in the manifest.
    pub sha256_written_to_manifest: bool,
}

fn build_client() -> Result<Client> {
    Client::builder()
        .user_agent(concat!("ddai-connectome/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(600))
        .build()
        .context("building HTTP client")
}

pub fn run_fetch(opts: &FetchOptions) -> Result<Vec<FetchOutcome>> {
    let manifest = Manifest::load(&opts.manifest_path)?;
    std::fs::create_dir_all(&opts.dest).with_context(|| format!("creating {}", opts.dest.display()))?;
    let client = build_client()?;

    manifest
        .files
        .iter()
        .map(|entry| fetch_one(&client, entry, &opts.dest, opts.update_sha256, &opts.manifest_path))
        .collect()
}

/// HEAD's the pinned URL and refuses (returns `Err`) unless the remote object's size,
/// generation and MD5 all still match `entry` exactly. This runs before every download attempt,
/// including when the final file is already present locally — it is the check that catches "the
/// bucket object was overwritten" even when we otherwise wouldn't touch the network at all.
fn head_check(client: &Client, entry: &FileEntry) -> Result<()> {
    let url = entry.pinned_url();
    let resp = client.head(&url).send().with_context(|| format!("HEAD {url}"))?;
    if !resp.status().is_success() {
        bail!(
            "{}: HEAD {url} returned {} — the pinned generation may no longer exist (object overwritten or deleted); refusing to fetch",
            entry.name,
            resp.status()
        );
    }

    let content_length = resp
        .content_length()
        .with_context(|| format!("{}: HEAD response has no Content-Length", entry.name))?;
    if content_length != entry.size {
        bail!(
            "{}: size mismatch: manifest says {} bytes, remote HEAD says {content_length}",
            entry.name,
            entry.size
        );
    }

    let generation_header = resp
        .headers()
        .get("x-goog-generation")
        .and_then(|v| v.to_str().ok())
        .with_context(|| format!("{}: HEAD response has no x-goog-generation header", entry.name))?;
    let remote_generation: u64 = generation_header.parse().with_context(|| {
        format!(
            "{}: x-goog-generation {generation_header:?} is not a number",
            entry.name
        )
    })?;
    if remote_generation != entry.generation {
        bail!(
            "{}: generation mismatch: manifest says {}, remote says {remote_generation} (pinned object was overwritten)",
            entry.name,
            entry.generation
        );
    }

    let hash_values: Vec<&str> = resp
        .headers()
        .get_all("x-goog-hash")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .collect();
    let remote_md5 = parse_goog_md5_hex(hash_values.iter().copied())
        .with_context(|| format!("{}: HEAD response has no parsable x-goog-hash md5", entry.name))?;
    if !remote_md5.eq_ignore_ascii_case(&entry.md5) {
        bail!(
            "{}: md5 mismatch: manifest says {}, remote HEAD says {remote_md5}",
            entry.name,
            entry.md5
        );
    }

    Ok(())
}

/// What a `.part` file's current length means relative to the pinned size, decided as a pure
/// function (no I/O) so it's unit-testable and [`download_or_resume`]'s branches read directly
/// off it instead of re-deriving the same three comparisons inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartState {
    /// No `.part` file, or it happens to be empty.
    Absent,
    /// `.part` is shorter than the pinned size: resume with a `Range` GET from here.
    Partial(u64),
    /// `.part` is already exactly the pinned size: nothing to download, just verify + rename.
    Complete,
    /// `.part` is longer than the pinned size: cannot mean a valid partial download of *this*
    /// pinned object (HEAD already confirmed the remote size matches the manifest) — discard and
    /// start over rather than guess where it came from.
    Oversized,
}

fn classify_part(existing_len: u64, target_size: u64) -> PartState {
    if existing_len == 0 {
        PartState::Absent
    } else if existing_len == target_size {
        PartState::Complete
    } else if existing_len < target_size {
        PartState::Partial(existing_len)
    } else {
        PartState::Oversized
    }
}

/// Outcome of [`download_or_resume`]: how many bytes were already in `.part` before this call,
/// and whether a GET actually happened this run (distinguishes "resumed/finished a download" from
/// "found an already-complete `.part` left over from a previous run" for logging/error messages).
#[derive(Debug, Clone, Copy)]
struct DownloadOutcome {
    resumed_from: u64,
    performed_get: bool,
}

/// Downloads (or resumes downloading) `entry` into `part_path`, leaving exactly `entry.size`
/// bytes in it on success.
fn download_or_resume(client: &Client, entry: &FileEntry, part_path: &Path) -> Result<DownloadOutcome> {
    let existing_len = if part_path.is_file() {
        std::fs::metadata(part_path)?.len()
    } else {
        0
    };

    let existing_len = match classify_part(existing_len, entry.size) {
        PartState::Oversized => {
            std::fs::remove_file(part_path)
                .with_context(|| format!("removing oversized partial download {}", part_path.display()))?;
            0
        }
        _ => existing_len,
    };

    if let PartState::Complete = classify_part(existing_len, entry.size) {
        // Fully downloaded already, just never renamed (e.g. crashed between write and rename).
        return Ok(DownloadOutcome {
            resumed_from: existing_len,
            performed_get: false,
        });
    }

    let url = entry.pinned_url();
    let mut request = client.get(&url);
    let mut file = if existing_len > 0 {
        request = request.header(RANGE, format!("bytes={existing_len}-"));
        OpenOptions::new()
            .append(true)
            .open(part_path)
            .with_context(|| format!("opening {}", part_path.display()))?
    } else {
        File::create(part_path).with_context(|| format!("creating {}", part_path.display()))?
    };

    let mut resp = request.send().with_context(|| format!("GET {url}"))?;
    if existing_len > 0 {
        if resp.status() != StatusCode::PARTIAL_CONTENT {
            bail!(
                "{}: expected 206 Partial Content resuming from byte {existing_len}, got {}",
                entry.name,
                resp.status()
            );
        }
    } else if !resp.status().is_success() {
        bail!("{}: GET {url} returned {}", entry.name, resp.status());
    }

    io::copy(&mut resp, &mut file)
        .with_context(|| format!("downloading {} into {}", entry.name, part_path.display()))?;
    file.flush()?;

    Ok(DownloadOutcome {
        resumed_from: existing_len,
        performed_get: true,
    })
}

/// Moves a `.part` that failed verification out of the way (to `<name>.part.bad`, overwriting
/// any previous one) so the *next* run doesn't see a same-length `.part`, re-verify it, and fail
/// again with a misleading "downloaded but failed verification" — without ever having downloaded
/// anything that run. Returns the quarantine path if the move succeeded (best-effort: if it
/// fails too, the original error is still returned to the caller, just without a quarantine path
/// to mention).
fn quarantine_bad_part(part_path: &Path, dest: &Path, name: &str) -> Option<PathBuf> {
    let bad_path = dest.join(format!("{name}.part.bad"));
    std::fs::rename(part_path, &bad_path).ok().map(|()| bad_path)
}

fn fetch_one(
    client: &Client,
    entry: &FileEntry,
    dest: &Path,
    update_sha256: bool,
    manifest_path: &Path,
) -> Result<FetchOutcome> {
    let final_path = dest.join(&entry.name);
    let part_path = dest.join(format!("{}.part", entry.name));

    head_check(client, entry)?;

    if final_path.is_file() {
        let hashes = verify_local_file(&final_path, entry).with_context(|| {
            format!(
                "{}: already present at {} but failed verification (tampered or corrupt?) — remove it and re-run fetch to redownload",
                entry.name,
                final_path.display()
            )
        })?;
        // Even when nothing needs downloading, `--update-sha256` must still be able to backfill
        // a still-empty pin (e.g. the user ran a plain `fetch` first and only now adds the
        // flag) — it must not require deleting and re-downloading the file just to compute a
        // hash we already just computed for verification.
        let sha256_written_to_manifest = maybe_write_sha256(entry, &hashes, update_sha256, manifest_path)?;
        return Ok(FetchOutcome {
            name: entry.name.clone(),
            action: FetchAction::AlreadyPresent,
            sha256_written_to_manifest,
        });
    }

    let download = download_or_resume(client, entry, &part_path)?;
    let hashes = match verify_local_file(&part_path, entry) {
        Ok(hashes) => hashes,
        Err(verify_err) => {
            let what = if download.performed_get {
                "downloaded"
            } else {
                "found an already-complete .part from a previous run"
            };
            let verify_err = anyhow::Error::new(verify_err);
            return Err(match quarantine_bad_part(&part_path, dest, &entry.name) {
                Some(bad_path) => verify_err.context(format!(
                    "{}: {what}, but failed verification — moved the bad partial download to {} \
                     for inspection; re-run fetch to redownload from scratch",
                    entry.name,
                    bad_path.display()
                )),
                None => verify_err.context(format!(
                    "{}: {what}, but failed verification, AND failed to move {} out of the way — \
                     remove it by hand before re-running fetch, or it will fail again the same way",
                    entry.name,
                    part_path.display()
                )),
            });
        }
    };

    let sha256_written_to_manifest = maybe_write_sha256(entry, &hashes, update_sha256, manifest_path)?;

    std::fs::rename(&part_path, &final_path)
        .with_context(|| format!("renaming {} to {}", part_path.display(), final_path.display()))?;

    Ok(FetchOutcome {
        name: entry.name.clone(),
        action: FetchAction::Downloaded {
            resumed_from: download.resumed_from,
            performed_get: download.performed_get,
            hashes,
        },
        sha256_written_to_manifest,
    })
}

/// Writes `hashes.sha256_hex` into the manifest for `entry`, but only if `update_sha256` was
/// requested and `entry` doesn't already have one pinned (never overwrites a pin).
fn maybe_write_sha256(
    entry: &FileEntry,
    hashes: &FileHashes,
    update_sha256: bool,
    manifest_path: &Path,
) -> Result<bool> {
    if update_sha256 && !entry.has_sha256() {
        update_sha256_in_file(manifest_path, &entry.name, &hashes.sha256_hex)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, sha256: &str) -> FileEntry {
        FileEntry {
            name: name.to_string(),
            url: format!("https://example.com/{name}"),
            generation: 1,
            size: 10,
            md5: "aa".to_string(),
            sha256: sha256.to_string(),
            license: "CC-BY-4.0".into(),
            source_version: "v1".into(),
            citation: "x".into(),
        }
    }

    fn hashes() -> FileHashes {
        FileHashes {
            size: 10,
            md5_hex: "aa".to_string(),
            sha256_hex: "deadbeef".to_string(),
        }
    }

    const MANIFEST: &str = "[[files]]\nname = \"f.feather\"\nsha256 = \"\"\n";

    #[test]
    fn maybe_write_sha256_no_op_without_the_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.toml");
        std::fs::write(&path, MANIFEST).unwrap();
        let e = entry("f.feather", "");
        let written = maybe_write_sha256(&e, &hashes(), false, &path).unwrap();
        assert!(!written);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), MANIFEST);
    }

    #[test]
    fn maybe_write_sha256_writes_when_flag_set_and_entry_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.toml");
        std::fs::write(&path, MANIFEST).unwrap();
        let e = entry("f.feather", "");
        let written = maybe_write_sha256(&e, &hashes(), true, &path).unwrap();
        assert!(written);
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("sha256 = \"deadbeef\"")
        );
    }

    #[test]
    fn maybe_write_sha256_never_overwrites_an_existing_pin() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.toml");
        std::fs::write(&path, MANIFEST).unwrap();
        let e = entry("f.feather", "already-pinned");
        let written = maybe_write_sha256(&e, &hashes(), true, &path).unwrap();
        assert!(!written, "entry already has a sha256 pinned, must not be touched");
        // File on disk is untouched too (the entry we constructed doesn't match what's on disk,
        // but the point stands: `has_sha256()` on the in-memory entry alone gates the write).
        assert_eq!(std::fs::read_to_string(&path).unwrap(), MANIFEST);
    }

    #[test]
    fn classify_part_covers_all_four_cases() {
        assert_eq!(classify_part(0, 100), PartState::Absent);
        assert_eq!(classify_part(40, 100), PartState::Partial(40));
        assert_eq!(classify_part(100, 100), PartState::Complete);
        assert_eq!(classify_part(150, 100), PartState::Oversized);
        // A pinned size of 0 is a degenerate case that should never occur for a real file
        // (nothing in `manifests/connectome.toml` is empty). `existing_len == 0` classifies as
        // `Absent` regardless of the target, not `Complete`: it's simpler and, for this one
        // never-really-happens case, arguably more correct too — `Absent` still drives a GET
        // (degenerate, but at least it talks to the server), whereas `Complete` would skip
        // straight to verification without ever having fetched anything for a file that was
        // never actually confirmed against the remote.
        assert_eq!(classify_part(0, 0), PartState::Absent);
    }

    #[test]
    fn quarantine_bad_part_moves_the_file_and_returns_its_new_path() {
        let dir = tempfile::tempdir().unwrap();
        let part_path = dir.path().join("f.feather.part");
        std::fs::write(&part_path, b"corrupt bytes").unwrap();

        let bad_path = quarantine_bad_part(&part_path, dir.path(), "f.feather").unwrap();
        assert_eq!(bad_path, dir.path().join("f.feather.part.bad"));
        assert!(!part_path.exists(), "the .part must be gone from its original path");
        assert!(bad_path.is_file());
        assert_eq!(std::fs::read(&bad_path).unwrap(), b"corrupt bytes");
    }

    #[test]
    fn quarantine_bad_part_overwrites_a_previous_bad_file() {
        let dir = tempfile::tempdir().unwrap();
        let part_path = dir.path().join("f.feather.part");
        let bad_path = dir.path().join("f.feather.part.bad");
        std::fs::write(&bad_path, b"stale from an earlier failed run").unwrap();
        std::fs::write(&part_path, b"newly corrupt bytes").unwrap();

        let returned = quarantine_bad_part(&part_path, dir.path(), "f.feather").unwrap();
        assert_eq!(returned, bad_path);
        assert_eq!(std::fs::read(&bad_path).unwrap(), b"newly corrupt bytes");
    }

    #[test]
    fn quarantine_bad_part_returns_none_when_the_part_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let part_path = dir.path().join("does-not-exist.part");
        assert_eq!(quarantine_bad_part(&part_path, dir.path(), "f.feather"), None);
    }

    #[test]
    fn a_corrupted_complete_part_is_quarantined_and_does_not_get_stuck() {
        // Regression for the bug this whole module was fixed for: a `.part` that already has
        // exactly the pinned size, but doesn't verify (simulating a corrupted download that
        // completed before the process died) must not be left in place — a second call with the
        // same setup must see it gone (moved to .bad), not hit the exact same dead end again.
        let dir = tempfile::tempdir().unwrap();
        let entry = entry("f.feather", "");
        let part_path = dir.path().join("f.feather.part");
        // Wrong content, but the right *length* (`entry.size` is 10 from the `entry` helper).
        std::fs::write(&part_path, b"wrongbytes").unwrap();
        assert_eq!(std::fs::metadata(&part_path).unwrap().len(), entry.size);

        assert!(
            verify_local_file(&part_path, &entry).is_err(),
            "sanity: this content must fail verification"
        );
        let bad_path = quarantine_bad_part(&part_path, dir.path(), &entry.name).unwrap();

        assert!(!part_path.exists());
        assert!(bad_path.is_file());
    }
}
