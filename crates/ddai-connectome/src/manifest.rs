//! `manifests/connectome.toml`: the pinned file list `fetch` downloads and verifies against.
//! See that file's header comment for what each field means.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One pinned remote file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub url: String,
    pub generation: u64,
    pub size: u64,
    pub md5: String,
    /// Empty string means "not pinned yet"; see `manifests/connectome.toml`'s header comment.
    #[serde(default)]
    pub sha256: String,
    pub license: String,
    pub source_version: String,
    pub citation: String,
}

impl FileEntry {
    pub fn has_sha256(&self) -> bool {
        !self.sha256.is_empty()
    }

    /// The full pinned download URL: `{url}?generation={generation}`.
    pub fn pinned_url(&self) -> String {
        format!("{}?generation={}", self.url, self.generation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    #[serde(rename = "files")]
    pub files: Vec<FileEntry>,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading manifest {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing manifest {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        Ok(toml::from_str(text)?)
    }

    pub fn find(&self, name: &str) -> Option<&FileEntry> {
        self.files.iter().find(|f| f.name == name)
    }
}

/// Writes `sha256` into `path`'s text for the `[[files]]` entry named `file_name`, but only if
/// that entry's current `sha256` is empty (`""`) — matching `fetch --update-sha256`'s contract
/// of never overwriting an already-pinned hash. Returns whether the file was modified.
///
/// This edits the raw TOML text rather than parsing into a `Manifest` and re-serializing it, so
/// every comment, blank line and the ordering/formatting of every *other* entry survives
/// byte-for-byte — only the one `sha256 = "..."` line changes.
pub fn update_sha256_in_file(path: &Path, file_name: &str, sha256: &str) -> Result<bool> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading manifest {}", path.display()))?;
    let (new_text, changed) = update_sha256_in_text(&text, file_name, sha256);
    if changed {
        std::fs::write(path, new_text).with_context(|| format!("writing manifest {}", path.display()))?;
    }
    Ok(changed)
}

/// Pure text transform behind [`update_sha256_in_file`] (kept separate so it can be unit tested
/// without touching the filesystem).
fn update_sha256_in_text(text: &str, file_name: &str, sha256: &str) -> (String, bool) {
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let mut in_matching_entry = false;
    let mut changed = false;

    for line in &mut lines {
        let trimmed = line.trim();
        if trimmed == "[[files]]" {
            in_matching_entry = false;
            continue;
        }
        if let Some(value) = parse_key_string(trimmed, "name") {
            in_matching_entry = value == file_name;
            continue;
        }
        if in_matching_entry && let Some(current) = parse_key_string(trimmed, "sha256") {
            if current.is_empty() {
                *line = format!("sha256 = \"{sha256}\"");
                changed = true;
            }
            // Only one `sha256` line per entry; stop looking once we've seen it so a
            // later, unrelated `[[files]]` block can never be mistaken for this one.
            in_matching_entry = false;
        }
    }

    let mut new_text = lines.join("\n");
    if text.ends_with('\n') {
        new_text.push('\n');
    }
    (new_text, changed)
}

/// Parses a simple, single-line, unescaped-string TOML assignment `key = "value"` (which is all
/// `manifests/connectome.toml` ever uses for `name` and `sha256`). Returns `None` if `line`
/// isn't such an assignment for `key`.
fn parse_key_string(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=')?;
    let rest = rest.trim();
    let inner = rest.strip_prefix('"')?.strip_suffix('"')?;
    Some(inner.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"# header comment
version = 1

[[files]]
name = "a.feather"
url = "https://example.com/a.feather"
generation = 1
size = 10
md5 = "aa"
sha256 = ""
license = "CC-BY-4.0"
source_version = "v1"
citation = "x"

[[files]]
name = "b.feather"
url = "https://example.com/b.feather"
generation = 2
size = 20
md5 = "bb"
sha256 = "already-pinned"
license = "CC-BY-4.0"
source_version = "v1"
citation = "y"
"#;

    #[test]
    fn parses_manifest_with_mixed_sha256() {
        let m = Manifest::parse(SAMPLE).unwrap();
        assert_eq!(m.version, 1);
        assert_eq!(m.files.len(), 2);
        let a = m.find("a.feather").unwrap();
        assert!(!a.has_sha256());
        assert_eq!(a.generation, 1);
        assert_eq!(a.pinned_url(), "https://example.com/a.feather?generation=1");
        let b = m.find("b.feather").unwrap();
        assert!(b.has_sha256());
        assert_eq!(b.sha256, "already-pinned");
    }

    #[test]
    fn find_returns_none_for_unknown_name() {
        let m = Manifest::parse(SAMPLE).unwrap();
        assert!(m.find("nope.feather").is_none());
    }

    #[test]
    fn update_fills_empty_sha256_for_named_entry_only() {
        let (new_text, changed) = update_sha256_in_text(SAMPLE, "a.feather", "deadbeef");
        assert!(changed);
        let m = Manifest::parse(&new_text).unwrap();
        assert_eq!(m.find("a.feather").unwrap().sha256, "deadbeef");
        // The other entry, and every other field of the edited entry, are untouched.
        assert_eq!(m.find("b.feather").unwrap().sha256, "already-pinned");
        assert_eq!(m.find("a.feather").unwrap().md5, "aa");
        assert_eq!(m.find("a.feather").unwrap().citation, "x");
        // Only the one line changed; everything else (including the header comment and
        // blank lines) is byte-for-byte identical.
        let orig_lines: Vec<&str> = SAMPLE.lines().collect();
        let new_lines: Vec<&str> = new_text.lines().collect();
        assert_eq!(orig_lines.len(), new_lines.len());
        let diff_count = orig_lines.iter().zip(new_lines.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(diff_count, 1);
        assert!(new_text.contains("# header comment"));
    }

    #[test]
    fn update_refuses_to_overwrite_existing_sha256() {
        let (new_text, changed) = update_sha256_in_text(SAMPLE, "b.feather", "some-other-hash");
        assert!(!changed);
        assert_eq!(new_text, SAMPLE); // byte-for-byte unchanged
        let m = Manifest::parse(&new_text).unwrap();
        assert_eq!(m.find("b.feather").unwrap().sha256, "already-pinned");
    }

    #[test]
    fn update_is_a_no_op_for_unknown_file_name() {
        let (new_text, changed) = update_sha256_in_text(SAMPLE, "does-not-exist.feather", "x");
        assert!(!changed);
        assert_eq!(new_text, SAMPLE);
    }

    #[test]
    fn update_sha256_in_file_round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("connectome.toml");
        std::fs::write(&path, SAMPLE).unwrap();

        let changed = update_sha256_in_file(&path, "a.feather", "deadbeef").unwrap();
        assert!(changed);
        let reloaded = Manifest::load(&path).unwrap();
        assert_eq!(reloaded.find("a.feather").unwrap().sha256, "deadbeef");

        // Second call with the now-filled entry must be a no-op (never overwrite a pin).
        let changed_again = update_sha256_in_file(&path, "a.feather", "different-hash").unwrap();
        assert!(!changed_again);
        let reloaded_again = Manifest::load(&path).unwrap();
        assert_eq!(reloaded_again.find("a.feather").unwrap().sha256, "deadbeef");
    }
}
