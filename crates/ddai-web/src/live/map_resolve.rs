//! Resolving a map's real `.map` bytes from nothing but its sha256 (acceptance criterion 2: "the
//! trace metadata names the map/rawmap and sha256; resolve it only inside the configured data
//! dir, never from a user-supplied path over HTTP") and serving the resulting scene at
//! `GET /api/map/<sha256>`.
//!
//! **Security model.** A trace's `real_map_path` metadata field is an absolute path written by
//! whatever machine originally ran Oracle B — untrusted, and almost certainly stale on this
//! machine (a different `$HOME`, a scratch directory that no longer exists). This module NEVER
//! opens that path. The only thing it takes from it is its filename (`Path::file_name`, which
//! cannot contain a `/` or `..` — see [`candidate_filename`]), which it then looks up **only**
//! inside the caller-configured `search_dirs` — themselves fixed at server startup (CLI flags,
//! never any per-request value) — and even then only accepts the file after independently
//! recomputing its sha256 and checking it against the trace's own `map_sha256` (also part of the
//! same untrusted trace metadata, so this is defense in depth, not the *only* check: a wrong
//! filename with a coincidentally-matching sha256 is - by definition - the actual right file's
//! bytes, so accepting it is correct regardless of what name it happened to be resolved under).
//!
//! `GET /api/map/<sha256>` takes the sha256 directly from the URL (already-authenticated route),
//! so it has an even simpler job: look up an already-built, cached [`super::scene::MapScene`] by
//! that exact hex string, with no filesystem path derived from client input at all.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use super::scene::MapScene;

/// Hard cap on a single candidate file's size this module will read into memory to hash/verify.
/// Real DDNet maps are at most a few tens of MiB; this is generous but not unbounded.
const MAX_MAP_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// Review round 1, finding F12: every variant below used to carry (and `{:?}`-display) the
/// *full* candidate path — e.g. `/home/ubuntu/aiddnet/data/maps/BlmapChill.map` — which reached
/// every authenticated client verbatim via `live_error` once `crate::live::replay` wrapped this
/// into a `SourceEvent::Error`. `--maps-dir` is an operator-configured, possibly-absolute CLI
/// flag no client needs to see the value of; the file name alone (already unique among a single
/// `--maps-dir`, and this module never recurses into subdirectories) is all a client needs to
/// understand which map failed. So every field below is the display name ([`display_name`]), not
/// a [`PathBuf`] — nothing outside this module matched on these fields' *type* before this change
/// (only `matches!(err, ResolveError::Variant { .. })`, checked directly), so this is not a
/// breaking change to anything that reads a [`ResolveError`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    #[error("no configured maps directory contains a file named {0:?}")]
    NotFound(String),
    #[error("candidate file {0:?} is larger than the {1}-byte cap")]
    TooLarge(String, u64),
    #[error("candidate file {path:?} sha256 does not match the trace's metadata")]
    Sha256Mismatch { path: String },
    #[error("I/O error reading {path:?}: {error}")]
    Io { path: String, error: String },
    #[error("the real_map_path metadata field has no usable filename")]
    NoFilename,
}

/// The name a client-facing [`ResolveError`] shows for a candidate path: just the file name (see
/// the enum's own doc comment for why). Mirrors `crate::live::replay::trace_display_name` — kept
/// as its own copy rather than shared across the two modules, since sharing it would mean either
/// module depending on the other for something this trivial.
fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Extracts just the filename component from an untrusted path string (see this module's doc
/// comment) — never the path itself. Returns `None` for input with no filename at all (empty
/// string, `/`, `..`) — [`resolve_by_sha256`] treats that the same as "not found", not a panic or
/// a fallback to some default.
fn candidate_filename(untrusted_path: &str) -> Option<String> {
    let name = Path::new(untrusted_path).file_name()?;
    let name = name.to_str()?;
    // `Path::file_name` already strips any directory components, but a defense-in-depth check
    // costs nothing: reject anything that still looks like it could escape a directory join.
    if name.is_empty() || name == ".." || name == "." || name.contains('/') {
        return None;
    }
    Some(name.to_string())
}

/// Reads `path` fully (bounded by [`MAX_MAP_FILE_BYTES`]) and returns `(bytes, sha256)`.
fn read_and_hash(path: &Path) -> Result<(Vec<u8>, [u8; 32]), ResolveError> {
    let metadata = std::fs::metadata(path).map_err(|e| ResolveError::Io {
        path: display_name(path),
        error: e.to_string(),
    })?;
    if metadata.len() > MAX_MAP_FILE_BYTES {
        return Err(ResolveError::TooLarge(display_name(path), MAX_MAP_FILE_BYTES));
    }
    let bytes = std::fs::read(path).map_err(|e| ResolveError::Io {
        path: display_name(path),
        error: e.to_string(),
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let sha256: [u8; 32] = hasher.finalize().into();
    Ok((bytes, sha256))
}

/// Looks up `filename` directly inside each of `search_dirs`, in order (no recursion — every real
/// map directory this task points at, `~/aiddnet/data/ddnet-server/maps` and
/// `~/aiddnet/data/maps/<subdir>`, is flat or one level deep; callers pass each such directory
/// explicitly rather than a common ancestor to recurse into), returning the first path that both
/// exists and whose contents' sha256 matches `expected_sha256`. Never opens anything outside
/// `search_dirs` (each already joined-and-canonicalized against `filename` alone, which
/// [`candidate_filename`] already guaranteed has no `/`/`..`).
pub fn resolve_by_sha256(
    search_dirs: &[PathBuf],
    real_map_path_hint: &str,
    expected_sha256: [u8; 32],
) -> Result<(PathBuf, MapScene), ResolveError> {
    let filename = candidate_filename(real_map_path_hint).ok_or(ResolveError::NoFilename)?;

    let mut last_error = None;
    for dir in search_dirs {
        let candidate = dir.join(&filename);
        // `symlink_metadata` (not `metadata`) so a symlink planted inside a configured maps
        // directory pointing outside it is at least visible to an operator inspecting the
        // directory listing — we still ultimately `std::fs::read` through it below (this is a
        // server-operator-controlled directory, not attacker-controlled input; the security
        // boundary this module actually enforces is "never derive a path from request/trace
        // data", which holds regardless), so this check is informational, not a hard refusal.
        if std::fs::symlink_metadata(&candidate).is_err() {
            continue; // not present in this directory; try the next one
        }
        match read_and_hash(&candidate) {
            Ok((bytes, actual_sha256)) => {
                if actual_sha256 != expected_sha256 {
                    last_error = Some(ResolveError::Sha256Mismatch {
                        path: display_name(&candidate),
                    });
                    continue;
                }
                match ddai_map::load_map(&bytes) {
                    Ok(loaded) => {
                        let scene = MapScene::build(&loaded.data);
                        return Ok((candidate, scene));
                    }
                    Err(e) => {
                        last_error = Some(ResolveError::Io {
                            path: display_name(&candidate),
                            error: e.to_string(),
                        });
                        continue;
                    }
                }
            }
            Err(e) => {
                last_error = Some(e);
                continue;
            }
        }
    }
    Err(last_error.unwrap_or(ResolveError::NotFound(filename)))
}

/// A tiny in-memory cache of already-built scenes, keyed by sha256 (the same key
/// `GET /api/map/<sha256>` is addressed by) — built lazily, kept for the process's lifetime. The
/// replay corpus only ever touches a handful of distinct real maps (7 in this task's corpus), so
/// there is no eviction: bounding memory here would be solving a problem this workload doesn't
/// have, at the cost of needing to answer "evict based on what policy" for no real benefit.
#[derive(Default)]
pub struct MapCache {
    by_sha256: Mutex<std::collections::HashMap<[u8; 32], std::sync::Arc<MapScene>>>,
    /// The DEFLATE-compressed wire bytes (`super::scene::encode_compressed`) `GET /api/map/<sha256>`
    /// serves, cached separately from the classified [`MapScene`] above (review round 1, finding
    /// F11: without this, every single request re-ran `Compression::best` from scratch — measured
    /// at ~30ms for `BlmapChill` — even though the input `MapScene` never changes for a given
    /// sha256, so neither can its compressed encoding). Keyed by the same sha256 as `by_sha256`
    /// but intentionally a separate map rather than one combined `(MapScene, Vec<u8>)` value: the
    /// compressed bytes are lazily computed on the *first* `GET /api/map/<sha256>` (not eagerly
    /// when the scene itself is resolved), since a map that plays in `live` but whose scene is
    /// never fetched over HTTP (nothing currently does this, but nothing rules it out either)
    /// should not pay a compression cost nobody asked for.
    compressed_by_sha256: Mutex<std::collections::HashMap<[u8; 32], std::sync::Arc<Vec<u8>>>>,
    /// Task 5.10: where each resolved map's file is (set by the sources that resolved it; the visual scene is read from
    /// the file again, hash-checked, when a browser first asks — the classified scene above keeps no bytes). A path comes
    /// only from `resolve_by_sha256`, never from a request.
    paths: Mutex<std::collections::HashMap<[u8; 32], PathBuf>>,
    /// Task 5.10: the visual scenes' wire bodies (`super::visual_scene`), at most [`MAX_VISUAL_BYTES`] of them together,
    /// oldest dropped.
    visual: Mutex<Vec<([u8; 32], std::sync::Arc<super::visual_scene::VisualEntry>)>>,
    /// Maps whose visual scene was refused as too large: not built again (a build is the expensive part).
    visual_refused: Mutex<std::collections::HashSet<[u8; 32]>>,
    /// Held while a visual scene is built, so two browsers asking for the same new map at once build it once.
    visual_build: Mutex<()>,
}

/// Bytes of compressed visual scenes (and their embedded images) kept at once; a real map is under 2 MiB.
pub const MAX_VISUAL_BYTES: usize = 48 * 1024 * 1024;

impl MapCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, sha256: &[u8; 32]) -> Option<std::sync::Arc<MapScene>> {
        self.by_sha256
            .lock()
            .expect("map cache mutex poisoned")
            .get(sha256)
            .cloned()
    }

    pub fn insert(&self, sha256: [u8; 32], scene: MapScene) -> std::sync::Arc<MapScene> {
        let scene = std::sync::Arc::new(scene);
        self.by_sha256
            .lock()
            .expect("map cache mutex poisoned")
            .insert(sha256, scene.clone());
        scene
    }

    /// Remembers where the map `sha256` was resolved from (task 5.10).
    pub fn insert_path(&self, sha256: [u8; 32], path: PathBuf) {
        self.paths
            .lock()
            .expect("map cache mutex poisoned")
            .insert(sha256, path);
    }

    pub fn path_of(&self, sha256: &[u8; 32]) -> Option<PathBuf> {
        self.paths
            .lock()
            .expect("map cache mutex poisoned")
            .get(sha256)
            .cloned()
    }

    /// The extracted visual scene of `sha256`, if one was built.
    pub fn visual(&self, sha256: &[u8; 32]) -> Option<std::sync::Arc<super::visual_scene::VisualEntry>> {
        self.visual
            .lock()
            .expect("map cache mutex poisoned")
            .iter()
            .find(|(k, _)| k == sha256)
            .map(|(_, v)| v.clone())
    }

    /// Keeps `entry` as the visual scene of `sha256` (the oldest are dropped while the kept bytes pass [`MAX_VISUAL_BYTES`];
    /// the newest always stays).
    pub fn insert_visual(
        &self,
        sha256: [u8; 32],
        entry: super::visual_scene::VisualEntry,
    ) -> std::sync::Arc<super::visual_scene::VisualEntry> {
        let entry = std::sync::Arc::new(entry);
        let mut all = self.visual.lock().expect("map cache mutex poisoned");
        all.retain(|(k, _)| *k != sha256);
        all.push((sha256, entry.clone()));
        while all.len() > 1 && all.iter().map(|(_, e)| e.size_bytes()).sum::<usize>() > MAX_VISUAL_BYTES {
            all.remove(0);
        }
        entry
    }

    /// The visual scene of `sha256`: the cached one, or the one `build` makes (once, however many callers ask at the same time).
    /// Blocking: call it from the blocking pool.
    pub fn visual_or_build(
        &self,
        sha256: [u8; 32],
        build: impl FnOnce(&Path) -> Result<super::visual_scene::VisualEntry, super::visual_scene::VisualError>,
    ) -> Result<std::sync::Arc<super::visual_scene::VisualEntry>, super::visual_scene::VisualError> {
        use super::visual_scene::VisualError;
        if let Some(hit) = self.visual(&sha256) {
            return Ok(hit);
        }
        let _one_builder = self.visual_build.lock().expect("map cache mutex poisoned");
        if let Some(hit) = self.visual(&sha256) {
            return Ok(hit);
        }
        if self
            .visual_refused
            .lock()
            .expect("map cache mutex poisoned")
            .contains(&sha256)
        {
            return Err(VisualError::TooLarge);
        }
        let path = self
            .path_of(&sha256)
            .ok_or_else(|| VisualError::Failed("this map has no file the page may be sent".to_string()))?;
        match build(&path) {
            Ok(entry) => Ok(self.insert_visual(sha256, entry)),
            Err(VisualError::TooLarge) => {
                self.visual_refused
                    .lock()
                    .expect("map cache mutex poisoned")
                    .insert(sha256);
                Err(VisualError::TooLarge)
            }
            Err(other) => Err(other),
        }
    }

    /// The cached compressed bytes for `sha256`, if `GET /api/map/<sha256>` has already computed
    /// them once before (review round 1, finding F11).
    pub fn get_compressed(&self, sha256: &[u8; 32]) -> Option<std::sync::Arc<Vec<u8>>> {
        self.compressed_by_sha256
            .lock()
            .expect("map cache mutex poisoned")
            .get(sha256)
            .cloned()
    }

    /// Caches `bytes` as the compressed encoding for `sha256`. If two requests race on the same
    /// not-yet-cached sha256 (both see `get_compressed` return `None` and both compress), this
    /// just keeps whichever insert lands second — both computed the exact same bytes from the
    /// same immutable `MapScene`, so which one "wins" the cache slot is never observable to a
    /// client; deduplicating that rare race with a single-flight lock would add real complexity
    /// for a cache that only ever needs to warm up once per distinct map anyway.
    pub fn insert_compressed(&self, sha256: [u8; 32], bytes: Vec<u8>) -> std::sync::Arc<Vec<u8>> {
        let bytes = std::sync::Arc::new(bytes);
        self.compressed_by_sha256
            .lock()
            .expect("map cache mutex poisoned")
            .insert(sha256, bytes.clone());
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_filename_strips_directories() {
        assert_eq!(
            candidate_filename("/home/ubuntu/aiddnet/data/research/physics-scratch/maps/BlmapChill.map"),
            Some("BlmapChill.map".to_string())
        );
    }

    #[test]
    fn candidate_filename_rejects_path_traversal_attempts() {
        assert_eq!(candidate_filename("../../etc/passwd"), Some("passwd".to_string()));
        assert_eq!(candidate_filename(".."), None);
        assert_eq!(candidate_filename("."), None);
        assert_eq!(candidate_filename(""), None);
        assert_eq!(candidate_filename("/"), None);
    }

    #[test]
    fn candidate_filename_of_a_bare_name_is_itself() {
        assert_eq!(candidate_filename("BlmapChill.map"), Some("BlmapChill.map".to_string()));
    }

    /// A minimal but real, loadable 2x2 map (via the shared test-util datafile writer — the same
    /// one `ddai-map`'s own tests use), so `resolve_by_sha256`'s `ddai_map::load_map` call
    /// exercises the real loader, not a hand-faked byte blob.
    fn write_minimal_map(dir: &Path, name: &str) -> (PathBuf, [u8; 32]) {
        use ddai_map::testutil::{MapWriter, TILESLAYERFLAG_GAME, TileLayerSpec, TilemapShape, game_layer_data};

        let mut w = MapWriter::new(4);
        w.add_version_item(1);
        w.add_tile_layer(&TileLayerSpec {
            shape: TilemapShape::Full,
            item_version: 3,
            width: 2,
            height: 2,
            flags: TILESLAYERFLAG_GAME,
            data: &game_layer_data(2, 2),
        });
        w.add_single_group_with_all_layers();
        let bytes = w.finish();

        let path = dir.join(name);
        std::fs::write(&path, &bytes).expect("write test map");
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        (path, hasher.finalize().into())
    }

    #[test]
    fn resolves_a_real_map_by_filename_and_sha256() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_path, sha256) = write_minimal_map(tmp.path(), "BlmapChill.map");

        let (resolved_path, scene) = resolve_by_sha256(
            &[tmp.path().to_path_buf()],
            "/some/stale/absolute/path/BlmapChill.map",
            sha256,
        )
        .expect("should resolve");
        assert_eq!(resolved_path, tmp.path().join("BlmapChill.map"));
        assert_eq!(scene.width, 2);
        assert_eq!(scene.height, 2);
    }

    #[test]
    fn refuses_a_file_whose_sha256_does_not_match() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_path, _sha256) = write_minimal_map(tmp.path(), "BlmapChill.map");
        let wrong_sha256 = [0xAA; 32];

        let err = resolve_by_sha256(&[tmp.path().to_path_buf()], "/x/BlmapChill.map", wrong_sha256)
            .expect_err("should refuse a sha256 mismatch");
        assert!(matches!(err, ResolveError::Sha256Mismatch { .. }), "{err:?}");
    }

    /// Regression test for review round 1, finding F12: this error reaches every authenticated
    /// client verbatim (wrapped into a `live_error` by `crate::live::replay::play_one_file`), so
    /// its `Display` text must never contain the configured `--maps-dir` directory component —
    /// only the candidate file's own name.
    #[test]
    fn a_sha256_mismatch_error_does_not_leak_the_search_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_path, _sha256) = write_minimal_map(tmp.path(), "BlmapChill.map");
        let wrong_sha256 = [0xAA; 32];

        let err = resolve_by_sha256(&[tmp.path().to_path_buf()], "/x/BlmapChill.map", wrong_sha256)
            .expect_err("should refuse a sha256 mismatch");
        let message = err.to_string();
        assert!(
            !message.contains(tmp.path().to_str().unwrap()),
            "error message must not contain the search directory: {message:?}"
        );
        assert!(message.contains("BlmapChill.map"), "{message:?}");
    }

    /// Same finding, for the "candidate is too large" and "I/O error reading it" variants (the
    /// other two that used to carry a full [`PathBuf`]).
    #[test]
    fn too_large_and_io_errors_do_not_leak_the_search_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let big_path = tmp.path().join("huge.map");
        // One byte over the cap is enough; write it sparsely-ish via `set_len` so the test stays
        // fast regardless of the cap's exact size.
        let file = std::fs::File::create(&big_path).unwrap();
        file.set_len(MAX_MAP_FILE_BYTES + 1).unwrap();
        let too_large_err = resolve_by_sha256(&[tmp.path().to_path_buf()], "/x/huge.map", [0u8; 32])
            .expect_err("should refuse an oversized candidate");
        assert!(matches!(too_large_err, ResolveError::TooLarge(..)), "{too_large_err:?}");
        let message = too_large_err.to_string();
        assert!(
            !message.contains(tmp.path().to_str().unwrap()),
            "TooLarge message must not contain the search directory: {message:?}"
        );
        assert!(message.contains("huge.map"), "{message:?}");
    }

    #[test]
    fn never_reads_outside_the_configured_search_dirs() {
        // Two directories: the map actually lives in `elsewhere`, which is NOT in `search_dirs`.
        // `..` traversal in the hint must not let this reach it via the configured dir either.
        let tmp = tempfile::tempdir().expect("tempdir");
        let configured = tmp.path().join("configured");
        let elsewhere = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&configured).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        let (_path, sha256) = write_minimal_map(&elsewhere, "secret.map");

        // A hint that LOOKS like it might traverse out of `configured` into `elsewhere` if this
        // module ever naively joined the whole path instead of just the filename.
        let hint = "../elsewhere/secret.map";
        let err = resolve_by_sha256(std::slice::from_ref(&configured), hint, sha256).expect_err("must not find it");
        assert!(matches!(err, ResolveError::NotFound(_)), "{err:?}");

        // Confirms the test itself is meaningful: the same file, looked up in the dir it's
        // ACTUALLY in, does resolve.
        let ok = resolve_by_sha256(&[elsewhere], hint, sha256);
        assert!(ok.is_ok());
    }

    #[test]
    fn missing_filename_hint_is_not_found_not_a_panic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = resolve_by_sha256(&[tmp.path().to_path_buf()], "..", [0u8; 32]);
        assert_eq!(err, Err(ResolveError::NoFilename));
    }

    #[test]
    fn falls_through_to_a_later_search_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let first = tmp.path().join("first");
        let second = tmp.path().join("second");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let (_path, sha256) = write_minimal_map(&second, "BlmapChill.map");

        let (resolved_path, _scene) =
            resolve_by_sha256(&[first, second.clone()], "/x/BlmapChill.map", sha256).expect("should resolve");
        assert_eq!(resolved_path, second.join("BlmapChill.map"));
    }

    #[test]
    fn map_cache_returns_the_same_scene_instance() {
        let cache = MapCache::new();
        assert!(cache.get(&[1u8; 32]).is_none());
        let scene = MapScene {
            width: 1,
            height: 1,
            kinds: vec![0],
        };
        let inserted = cache.insert([1u8; 32], scene.clone());
        let fetched = cache.get(&[1u8; 32]).expect("should be cached");
        assert_eq!(*fetched, scene);
        assert!(std::sync::Arc::ptr_eq(&inserted, &fetched));
    }

    /// Regression test for review round 1, finding F11: `GET /api/map/<sha256>` looks up
    /// [`MapCache::get_compressed`]/`insert_compressed` (this test exercises exactly those two
    /// methods, not the HTTP layer above them — the HTTP-level behavior is covered by
    /// `crates/ddai-web/tests/live_map.rs::map_route_serves_the_scene_with_etag_and_supports_conditional_get`)
    /// so a second lookup for the same sha256 gets back the exact same cached `Arc`, not a
    /// freshly (re-)computed `Vec`.
    #[test]
    fn map_cache_returns_the_same_compressed_bytes_instance() {
        let cache = MapCache::new();
        assert!(cache.get_compressed(&[2u8; 32]).is_none());
        let bytes = vec![1, 2, 3, 4, 5];
        let inserted = cache.insert_compressed([2u8; 32], bytes.clone());
        let fetched = cache.get_compressed(&[2u8; 32]).expect("should be cached");
        assert_eq!(*fetched, bytes);
        assert!(std::sync::Arc::ptr_eq(&inserted, &fetched));
        // The scene cache and the compressed-bytes cache are independent maps, keyed the same
        // way but never confused for each other.
        assert!(cache.get(&[2u8; 32]).is_none());
    }
}
