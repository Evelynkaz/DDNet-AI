//! Privacy audit of a built dataset against the demos it came from (D-028, D-040): collects every
//! nickname and clan that appears in the demos' `ClientInfo` (plus the demo file names, which carry
//! nicknames in the real archives) and searches all output files for them: JSON files for a string
//! *value* equal to a name (keys are the tool's own words, a player may well be called "rank") and
//! for long names anywhere in the text, chunk files after decompression for any occurrence, and the
//! raw map files separately (byte-identical copies of the maps embedded in the demos, so nothing the
//! pipeline derived from a player can be in them; a coincidence with a map's own text is reported
//! as a map hit, not a leak). The names themselves never leave this module: the
//! report holds counts, lengths and indices only, so that it can be pasted anywhere.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_demo::Demo;
use ddai_net::view::View;

use crate::dataset::DatasetError;
use crate::run::discover;

/// Names shorter than this many bytes are not searched in binary data: a three-letter name matches
/// by chance.
pub const MIN_BINARY_NAME_LEN: usize = 4;
/// In JSON text a name is searched as a substring only from this length on (shorter ones are
/// searched as whole string values): "demo" is a key of the manifest, not a leak.
pub const MIN_JSON_SUBSTRING_LEN: usize = 8;

/// Where a name was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitKind {
    Json,
    Chunk,
    /// A raw map file (see the module docs).
    Map,
}

/// One name found in an output file. The name is identified by its index in the audit's (sorted)
/// list and its length, never by its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    pub kind: HitKind,
    /// Path of the file relative to the dataset directory.
    pub file: String,
    pub name_index: usize,
    pub name_len: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrivacyReport {
    /// Distinct nicknames, clans and demo file names searched for.
    pub names: usize,
    /// Of those, names too short for the binary search.
    pub short_names: usize,
    pub demos_read: usize,
    pub files_checked: usize,
    /// Bytes searched (chunks counted after decompression).
    pub bytes_checked: u64,
    pub hits: Vec<Hit>,
}

impl PrivacyReport {
    /// Hits in files derived from the demos (everything but the raw maps): a leak.
    pub fn leaks(&self) -> usize {
        self.hits.iter().filter(|h| h.kind != HitKind::Map).count()
    }
}

/// An error on a demo file: named by its index in the sorted listing, never by its path (the file
/// names carry nicknames, and this tool's output is meant to be pasted).
fn demo_io(index: usize) -> impl FnOnce(std::io::Error) -> DatasetError {
    move |source| DatasetError::Demo {
        what: format!("demo file #{index} (in sorted order)"),
        source,
    }
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> DatasetError + '_ {
    |source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Every nickname and clan in the `ClientInfo` items of the demo's snapshots.
pub fn names_in_demo(demo: &Demo<'_>) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut prev: Option<Arc<ddai_net::snapshot::Snapshot>> = None;
    for tick in demo.ticks() {
        let Ok(tick) = tick else { break };
        let Some(snap) = &tick.snapshot else { continue };
        if prev.as_ref().is_some_and(|p| Arc::ptr_eq(p, snap)) {
            continue;
        }
        prev = Some(Arc::clone(snap));
        for p in View::new(snap).players() {
            if let Some(ci) = p.client_info {
                for s in [ci.name, ci.clan] {
                    if !s.is_empty() {
                        names.insert(s);
                    }
                }
            }
        }
    }
    names
}

/// Collects the names of every `*.demo` under `demos_dir`: `ClientInfo` names and clans, file
/// names and file stems. Returns the sorted set and the number of demos read.
pub fn collect_names(demos_dir: &Path) -> Result<(BTreeSet<String>, usize), DatasetError> {
    let mut names = BTreeSet::new();
    let mut read = 0;
    for (index, f) in discover(demos_dir, "demo")?.into_iter().enumerate() {
        if let Some(n) = f.file_name() {
            names.insert(n.to_string_lossy().into_owned());
        }
        if let Some(n) = f.file_stem() {
            names.insert(n.to_string_lossy().into_owned());
        }
        let bytes = fs::read(&f).map_err(demo_io(index))?;
        if let Ok(demo) = Demo::parse(&bytes) {
            names.extend(names_in_demo(&demo));
            read += 1;
        }
    }
    names.retain(|n| !n.is_empty());
    Ok((names, read))
}

/// Counts every JSON string *value* (object keys are skipped).
fn string_values<'a>(v: &'a serde_json::Value, out: &mut std::collections::HashMap<&'a str, usize>) {
    match v {
        serde_json::Value::String(s) => *out.entry(s.as_str()).or_default() += 1,
        serde_json::Value::Array(a) => a.iter().for_each(|x| string_values(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| string_values(x, out)),
        _ => {}
    }
}

/// Names of at least [`MIN_BINARY_NAME_LEN`] bytes, indexed by their first four bytes so that a scan
/// of a large file costs one hash lookup per position, not one comparison per name.
struct PrefixIndex<'a> {
    by_prefix: std::collections::HashMap<[u8; 4], Vec<(usize, &'a [u8])>>,
}

impl<'a> PrefixIndex<'a> {
    fn new(names: &[&'a String], min_len: usize) -> Self {
        let mut by_prefix: std::collections::HashMap<[u8; 4], Vec<(usize, &'a [u8])>> =
            std::collections::HashMap::new();
        for (i, n) in names.iter().enumerate() {
            let b = n.as_bytes();
            if b.len() >= min_len.max(MIN_BINARY_NAME_LEN) {
                by_prefix.entry([b[0], b[1], b[2], b[3]]).or_default().push((i, b));
            }
        }
        PrefixIndex { by_prefix }
    }

    /// Occurrences of each indexed name in `bytes`, by name index.
    fn scan(&self, bytes: &[u8]) -> std::collections::BTreeMap<usize, usize> {
        let mut found = std::collections::BTreeMap::new();
        for (pos, w) in bytes.windows(MIN_BINARY_NAME_LEN).enumerate() {
            let Some(cands) = self.by_prefix.get(&[w[0], w[1], w[2], w[3]]) else {
                continue;
            };
            for &(i, name) in cands {
                if bytes[pos..].starts_with(name) {
                    *found.entry(i).or_default() += 1;
                }
            }
        }
        found
    }
}

/// Searches every file of the dataset directory for `names`.
pub fn scan(dataset_dir: &Path, names: &BTreeSet<String>) -> Result<PrivacyReport, DatasetError> {
    let list: Vec<&String> = names.iter().collect();
    let mut rep = PrivacyReport {
        names: list.len(),
        short_names: list.iter().filter(|n| n.len() < MIN_BINARY_NAME_LEN).count(),
        ..PrivacyReport::default()
    };
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![dataset_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).map_err(io(&d))? {
            let p = e.map_err(io(&d))?.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    let index = PrefixIndex::new(&list, MIN_BINARY_NAME_LEN);
    let json_index = PrefixIndex::new(&list, MIN_JSON_SUBSTRING_LEN);
    for p in files {
        let rel = p.strip_prefix(dataset_dir).unwrap_or(&p).display().to_string();
        let mut bytes = fs::read(&p).map_err(io(&p))?;
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.ends_with(".pf.zst") {
            bytes = zstd::decode_all(bytes.as_slice()).map_err(io(&p))?;
        }
        let kind = if name.ends_with(".json") {
            HitKind::Json
        } else if name.ends_with(".map") {
            HitKind::Map
        } else {
            HitKind::Chunk
        };
        rep.files_checked += 1;
        rep.bytes_checked += bytes.len() as u64;
        let found: std::collections::BTreeMap<usize, usize> = if kind == HitKind::Json {
            let mut found = json_index.scan(&bytes);
            let parsed = serde_json::from_slice::<serde_json::Value>(&bytes).ok();
            let mut values = std::collections::HashMap::new();
            if let Some(v) = &parsed {
                string_values(v, &mut values);
            }
            for (i, n) in list.iter().enumerate() {
                if let Some(&c) = values.get(n.as_str()) {
                    found.insert(i, c.max(found.get(&i).copied().unwrap_or(0)));
                }
            }
            found
        } else {
            index.scan(&bytes)
        };
        for (i, c) in found {
            rep.hits.push(Hit {
                kind,
                file: rel.clone(),
                name_index: i,
                name_len: list[i].len(),
                count: c,
            });
        }
    }
    Ok(rep)
}

/// Audits `dataset_dir` against the demos in `demos_dir`.
pub fn check(demos_dir: &Path, dataset_dir: &Path) -> Result<PrivacyReport, DatasetError> {
    let (names, read) = collect_names(demos_dir)?;
    let mut rep = scan(dataset_dir, &names)?;
    rep.demos_read = read;
    Ok(rep)
}

/// One line per finding, names never included.
pub fn render(rep: &PrivacyReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "privacy audit: {} names (nicknames, clans, demo file names; {} too short for the binary search) from {} demos; {} files, {} bytes searched; {} leaks, {} coincidences with raw map text",
        rep.names,
        rep.short_names,
        rep.demos_read,
        rep.files_checked,
        rep.bytes_checked,
        rep.leaks(),
        rep.hits.len() - rep.leaks()
    );
    for h in &rep.hits {
        let _ = writeln!(
            s,
            "  {} {}: name #{} ({} bytes) x{}",
            if h.kind == HitKind::Map { "map-text" } else { "LEAK" },
            h.file,
            h.name_index,
            h.name_len,
            h.count
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_in_any_kind_of_file_is_found_without_being_reported() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("chunks")).unwrap();
        fs::write(
            dir.path().join("report.json"),
            br#"{"a":"clean","tiny":"ab","keyname":1,"abc":2,"list":["xx abcdefgh yy"]}"#,
        )
        .unwrap();
        let packed = zstd::encode_all(&b"\x01\x02ZzSecretNick\x03"[..], 1).unwrap();
        fs::write(dir.path().join("chunks/00000.pf.zst"), packed).unwrap();
        let names: BTreeSet<String> = ["ZzSecretNick", "NeverThere", "ab", "abc", "abcdefgh", "keyname"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rep = scan(dir.path(), &names).unwrap();
        assert_eq!(rep.names, 6);
        assert_eq!(rep.short_names, 2);
        assert_eq!(rep.files_checked, 2);
        // The long name inside the decompressed chunk, the short one as a whole JSON string value,
        // the 8-byte one inside a JSON string; a name equal to a JSON key ("abc", "keyname") is not one.
        assert_eq!(rep.hits.len(), 3, "{:?}", rep.hits);
        assert_eq!(rep.leaks(), 3);
        assert!(
            rep.hits
                .iter()
                .any(|h| h.file.ends_with("00000.pf.zst") && h.name_len == 12)
        );
        assert!(rep.hits.iter().any(|h| h.file == "report.json" && h.name_len == 2));
        let text = render(&rep);
        assert!(!text.contains("ZzSecretNick") && !text.contains("NeverThere"));
    }

    #[test]
    fn a_clean_dataset_has_no_hits() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("manifest.json"), br#"{"demos":["player_1"]}"#).unwrap();
        let names: BTreeSet<String> = ["Alpha"].iter().map(|s| s.to_string()).collect();
        assert!(scan(dir.path(), &names).unwrap().hits.is_empty());
    }
}
