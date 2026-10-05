//! The teacher dataset on disk: chunked postcard + zstd files under `<dir>/chunks/` plus a JSON
//! manifest (`<dir>/manifest.json`), the layout of the human dataset (`ddai-dataset`) with
//! teacher-specific fields. Datasets *grow*: every DAgger round appends chunks tagged with its
//! round and the actor that played, and rewrites the manifest atomically, so an interrupted run
//! leaves a consistent dataset.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::types::{Episode, TEACHER_FORMAT, TEACHER_FORMAT_VERSION, TeacherChunk};

/// Roughly how many decisions go into one chunk file.
pub const CHUNK_TARGET_STEPS: usize = 40_000;

#[derive(Debug)]
pub struct StoreError(pub String);

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StoreError {}

fn io_err(path: &Path, e: impl std::fmt::Display) -> StoreError {
    StoreError(format!("{}: {e}", path.display()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArenaRef {
    pub name: String,
    /// sha256 of the map file (`None` for a synthetic arena).
    pub map_sha256: Option<String>,
    /// `train` or `holdout`.
    pub split: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkMeta {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    pub episodes: u32,
    pub steps: u64,
    /// DAgger round (`0` = the initial teacher games).
    pub round: u32,
    /// Who played the labelled slot (free text, e.g. `planner`, `fly@step12000`).
    pub actor: String,
    /// Free-text description of the opponents and mixing, for the record.
    pub setup: String,
    /// Identity of the collection job that wrote the chunk (its setup, base seed and game count);
    /// empty in datasets written before the markers existed. A job's chunks reach the manifest
    /// together (one atomic rewrite), so a key present in the manifest means the job finished.
    #[serde(default)]
    pub job_key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherManifest {
    pub format: String,
    pub format_version: u32,
    pub name: String,
    /// Commit of the code that wrote the first chunk.
    pub code_commit: String,
    pub arenas: Vec<ArenaRef>,
    pub chunks: Vec<ChunkMeta>,
    /// Rounds whose collection finished: every job of the round is in `chunks`. Datasets written
    /// before the marker existed have none; for those a round with chunks counts as complete
    /// ([`TeacherManifest::round_complete`]).
    #[serde(default)]
    pub rounds_complete: Vec<u32>,
}

impl TeacherManifest {
    /// Whether the collection of `round` finished. Legacy datasets (no job keys at all in the
    /// round's chunks, no markers) were only ever written by a collection that ran to the end.
    pub fn round_complete(&self, round: u32) -> bool {
        if self.rounds_complete.contains(&round) {
            return true;
        }
        let mut chunks = self.chunks.iter().filter(|c| c.round == round).peekable();
        chunks.peek().is_some() && chunks.all(|c| c.job_key.is_empty())
    }

    /// Job keys written into `round` that are not in `expected` (a resumed run whose job list changed:
    /// jobs of another length or seed are already in the dataset and would be collected next to the
    /// new ones, so the caller warns).
    pub fn unexpected_job_keys(&self, round: u32, expected: &[String]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for c in self.chunks.iter().filter(|c| c.round == round && !c.job_key.is_empty()) {
            if !expected.contains(&c.job_key) && !out.contains(&c.job_key) {
                out.push(c.job_key.clone());
            }
        }
        out
    }

    /// Whether the job `key` of `round` has its chunks in the dataset.
    pub fn job_done(&self, round: u32, key: &str) -> bool {
        !key.is_empty() && self.chunks.iter().any(|c| c.round == round && c.job_key == key)
    }

    pub fn total_steps(&self) -> u64 {
        self.chunks.iter().map(|c| c.steps).sum()
    }
    pub fn total_episodes(&self) -> u64 {
        self.chunks.iter().map(|c| u64::from(c.episodes)).sum()
    }
    /// Index of `name` in [`TeacherManifest::arenas`], adding it when missing.
    pub fn arena_index(&mut self, arena: ArenaRef) -> u16 {
        match self.arenas.iter().position(|a| a.name == arena.name) {
            Some(i) => i as u16,
            None => {
                self.arenas.push(arena);
                (self.arenas.len() - 1) as u16
            }
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A teacher dataset directory (created or opened).
pub struct TeacherStore {
    dir: PathBuf,
    pub manifest: TeacherManifest,
}

impl TeacherStore {
    pub fn create(dir: &Path, name: &str, code_commit: &str) -> Result<Self, StoreError> {
        std::fs::create_dir_all(dir.join("chunks")).map_err(|e| io_err(dir, e))?;
        let store = TeacherStore {
            dir: dir.to_path_buf(),
            manifest: TeacherManifest {
                format: TEACHER_FORMAT.to_string(),
                format_version: TEACHER_FORMAT_VERSION,
                name: name.to_string(),
                code_commit: code_commit.to_string(),
                arenas: Vec::new(),
                chunks: Vec::new(),
                rounds_complete: Vec::new(),
            },
        };
        store.write_manifest()?;
        Ok(store)
    }

    pub fn open(dir: &Path) -> Result<Self, StoreError> {
        let path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&path).map_err(|e| io_err(&path, e))?;
        let manifest: TeacherManifest = serde_json::from_str(&text).map_err(|e| io_err(&path, e))?;
        if manifest.format != TEACHER_FORMAT || manifest.format_version != TEACHER_FORMAT_VERSION {
            return Err(StoreError(format!(
                "{}: unsupported dataset format {:?} v{}",
                path.display(),
                manifest.format,
                manifest.format_version
            )));
        }
        Ok(TeacherStore {
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    /// Opens `dir` when it holds a manifest, else creates it.
    pub fn open_or_create(dir: &Path, name: &str, code_commit: &str) -> Result<Self, StoreError> {
        if dir.join("manifest.json").exists() {
            Self::open(dir)
        } else {
            Self::create(dir, name, code_commit)
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn write_manifest(&self) -> Result<(), StoreError> {
        let path = self.dir.join("manifest.json");
        let tmp = self.dir.join(format!("manifest.json.{}.tmp", std::process::id()));
        let text = serde_json::to_string_pretty(&self.manifest).map_err(|e| io_err(&path, e))?;
        std::fs::write(&tmp, text).map_err(|e| io_err(&tmp, e))?;
        std::fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))
    }

    /// Records that every job of `round` is in the dataset.
    pub fn mark_round_complete(&mut self, round: u32) -> Result<(), StoreError> {
        if !self.manifest.rounds_complete.contains(&round) {
            self.manifest.rounds_complete.push(round);
            self.write_manifest()?;
        }
        Ok(())
    }

    /// Writes `episodes` as chunks of about [`CHUNK_TARGET_STEPS`] decisions, tagged with `round`,
    /// `actor`, `setup` and the job's `job_key`, and updates the manifest (once, after every chunk
    /// file is in place). Returns the indices of the new chunks.
    pub fn append(
        &mut self,
        round: u32,
        actor: &str,
        setup: &str,
        job_key: &str,
        episodes: Vec<Episode>,
    ) -> Result<Vec<usize>, StoreError> {
        let mut new = Vec::new();
        let mut current: Vec<Episode> = Vec::new();
        let mut current_steps = 0usize;
        let mut groups: Vec<Vec<Episode>> = Vec::new();
        for ep in episodes {
            current_steps += ep.steps.len();
            current.push(ep);
            if current_steps >= CHUNK_TARGET_STEPS {
                groups.push(std::mem::take(&mut current));
                current_steps = 0;
            }
        }
        if !current.is_empty() {
            groups.push(current);
        }
        for group in groups {
            let index = self.manifest.chunks.len();
            let steps: u64 = group.iter().map(|e| e.steps.len() as u64).sum();
            let n_eps = group.len() as u32;
            let chunk = TeacherChunk {
                format_version: TEACHER_FORMAT_VERSION,
                episodes: group,
            };
            let raw = postcard::to_allocvec(&chunk).map_err(|e| StoreError(format!("encoding chunk: {e}")))?;
            let bytes = zstd::encode_all(raw.as_slice(), 5).map_err(|e| StoreError(format!("zstd: {e}")))?;
            let file = format!("chunks/{index:06}.bin.zst");
            let path = self.dir.join(&file);
            let tmp = self.dir.join(format!("{file}.{}.tmp", std::process::id()));
            std::fs::write(&tmp, &bytes).map_err(|e| io_err(&tmp, e))?;
            std::fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))?;
            self.manifest.chunks.push(ChunkMeta {
                file,
                sha256: hex(&Sha256::digest(&bytes)),
                bytes: bytes.len() as u64,
                episodes: n_eps,
                steps,
                round,
                actor: actor.to_string(),
                setup: setup.to_string(),
                job_key: job_key.to_string(),
            });
            new.push(index);
        }
        self.write_manifest()?;
        Ok(new)
    }

    /// Decodes chunk `index`, checking its sha256 against the manifest.
    pub fn read_chunk(&self, index: usize) -> Result<TeacherChunk, StoreError> {
        let meta = self
            .manifest
            .chunks
            .get(index)
            .ok_or_else(|| StoreError(format!("no chunk {index}")))?;
        let path = self.dir.join(&meta.file);
        let bytes = std::fs::read(&path).map_err(|e| io_err(&path, e))?;
        let actual = hex(&Sha256::digest(&bytes));
        if actual != meta.sha256 {
            return Err(StoreError(format!(
                "{}: sha256 {actual} does not match the manifest ({})",
                path.display(),
                meta.sha256
            )));
        }
        let raw = zstd::decode_all(bytes.as_slice()).map_err(|e| io_err(&path, e))?;
        let chunk: TeacherChunk = postcard::from_bytes(&raw).map_err(|e| io_err(&path, e))?;
        if chunk.format_version != TEACHER_FORMAT_VERSION {
            return Err(StoreError(format!(
                "{}: chunk format {}",
                path.display(),
                chunk.format_version
            )));
        }
        Ok(chunk)
    }

    /// Checks every chunk file against its recorded sha256 (and decodes it).
    pub fn verify(&self) -> Result<(), StoreError> {
        for i in 0..self.manifest.chunks.len() {
            self.read_chunk(i)?;
        }
        Ok(())
    }

    /// Indices of the chunks written in `round` (or all when `None`).
    pub fn chunks_of_round(&self, round: Option<u32>) -> Vec<usize> {
        self.manifest
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| round.is_none_or(|r| c.round == r))
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Outcome, TeacherStep, action_rec, char_rec};
    use ddai_brain::{Action, CharacterObservation};

    fn episode(seed: u64, n: usize) -> Episode {
        let step = |t: i32| TeacherStep {
            tick: t * 2,
            me: char_rec(&CharacterObservation::at_rest(0)),
            others: vec![char_rec(&CharacterObservation::at_rest(1))],
            target: 1,
            label: action_rec(&Action::neutral()),
            soft: None,
            played: action_rec(&Action::neutral()),
            flags: 0,
        };
        Episode {
            arena: 0,
            seed,
            players: 2,
            outcome: Outcome::Win,
            end_tick: 100,
            steps: (0..n as i32).map(step).collect(),
        }
    }

    #[test]
    fn append_round_trips_verifies_and_survives_reopening() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = TeacherStore::create(dir.path(), "t", "abc").unwrap();
        s.manifest.arena_index(ArenaRef {
            name: "pit".into(),
            map_sha256: None,
            split: "train".into(),
        });
        let eps = vec![episode(1, 30), episode(2, 40)];
        let new = s.append(0, "planner", "vs scripted", "", eps.clone()).unwrap();
        assert_eq!(new, vec![0]);
        let more = s.append(1, "fly@100", "mixed", "", vec![episode(3, 5)]).unwrap();
        assert_eq!(more, vec![1]);
        s.verify().unwrap();

        let back = TeacherStore::open(dir.path()).unwrap();
        assert_eq!(back.manifest, s.manifest);
        assert_eq!(back.manifest.total_steps(), 75);
        assert_eq!(back.manifest.total_episodes(), 3);
        assert_eq!(back.read_chunk(0).unwrap().episodes, eps);
        assert_eq!(back.chunks_of_round(Some(1)), vec![1]);
        assert_eq!(back.chunks_of_round(None), vec![0, 1]);
    }

    #[test]
    fn large_episode_lists_are_split_into_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = TeacherStore::create(dir.path(), "t", "abc").unwrap();
        let eps: Vec<Episode> = (0..10).map(|i| episode(i, CHUNK_TARGET_STEPS / 4)).collect();
        let new = s.append(0, "planner", "", "", eps).unwrap();
        assert!(new.len() >= 2, "{new:?}");
        assert_eq!(s.manifest.total_episodes(), 10);
    }

    #[test]
    fn a_corrupted_chunk_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = TeacherStore::create(dir.path(), "t", "abc").unwrap();
        s.append(0, "planner", "", "", vec![episode(1, 20)]).unwrap();
        let path = dir.path().join(&s.manifest.chunks[0].file);
        let mut bytes = std::fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x55;
        std::fs::write(&path, bytes).unwrap();
        assert!(s.read_chunk(0).unwrap_err().to_string().contains("sha256"));
        assert!(s.verify().is_err());
    }

    #[test]
    fn opening_a_foreign_manifest_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("manifest.json"),
            r#"{"format":"other","format_version":1,"name":"x","code_commit":"","arenas":[],"chunks":[]}"#,
        )
        .unwrap();
        assert!(TeacherStore::open(dir.path()).is_err());
    }

    #[test]
    fn job_keys_and_round_markers_tell_a_finished_round_from_a_partial_one() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = TeacherStore::create(dir.path(), "t", "abc").unwrap();
        assert!(!s.manifest.round_complete(1), "an empty round is not complete");
        s.append(
            1,
            "mlp:x",
            "pit vs scripted",
            "pit|seed=5|games=2",
            vec![episode(1, 10)],
        )
        .unwrap();
        assert!(s.manifest.job_done(1, "pit|seed=5|games=2"));
        assert!(
            !s.manifest.job_done(1, "pit|seed=6|games=2"),
            "another seed is another job"
        );
        assert!(
            !s.manifest.job_done(2, "pit|seed=5|games=2"),
            "another round is another job"
        );
        assert!(!s.manifest.job_done(1, ""), "the empty key is never a job");
        assert!(!s.manifest.round_complete(1), "one job of a round is not the round");
        s.mark_round_complete(1).unwrap();
        s.mark_round_complete(1).unwrap();
        assert!(TeacherStore::open(dir.path()).unwrap().manifest.round_complete(1));
        assert_eq!(s.manifest.rounds_complete, vec![1]);
    }

    #[test]
    fn unexpected_job_keys_name_the_jobs_a_changed_config_no_longer_lists() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = TeacherStore::create(dir.path(), "t", "abc").unwrap();
        s.append(1, "a", "x", "pit|seed=5|games=2|noise_len=2-6", vec![episode(1, 5)])
            .unwrap();
        s.append(1, "a", "x", "pit|seed=6|games=2|noise_len=2-6", vec![episode(2, 5)])
            .unwrap();
        s.append(2, "a", "x", "pit|seed=9|games=2|noise_len=2-6", vec![episode(3, 5)])
            .unwrap();
        let expected = vec!["pit|seed=5|games=2|noise_len=2-6".to_string()];
        assert_eq!(
            s.manifest.unexpected_job_keys(1, &expected),
            vec!["pit|seed=6|games=2|noise_len=2-6".to_string()]
        );
        assert!(s.manifest.unexpected_job_keys(3, &expected).is_empty());
        // Legacy chunks without a key are never reported.
        s.append(3, "a", "x", "", vec![episode(4, 5)]).unwrap();
        assert!(s.manifest.unexpected_job_keys(3, &expected).is_empty());
    }

    #[test]
    fn datasets_written_before_the_markers_count_their_rounds_as_complete() {
        // A manifest without `rounds_complete` and chunks without `job_key` (the E-005 datasets).
        let dir = tempfile::tempdir().unwrap();
        let mut s = TeacherStore::create(dir.path(), "t", "abc").unwrap();
        s.append(1, "fly:x", "pit", "", vec![episode(1, 10)]).unwrap();
        let text = std::fs::read_to_string(dir.path().join("manifest.json")).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v.as_object_mut().unwrap().remove("rounds_complete");
        for c in v["chunks"].as_array_mut().unwrap() {
            c.as_object_mut().unwrap().remove("job_key");
        }
        std::fs::write(dir.path().join("manifest.json"), v.to_string()).unwrap();
        let old = TeacherStore::open(dir.path()).unwrap();
        assert!(old.manifest.round_complete(1));
        assert!(!old.manifest.round_complete(2));
    }
}
