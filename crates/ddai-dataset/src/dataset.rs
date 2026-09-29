//! The on-disk dataset: chunked postcard + zstd files, a manifest, a skill table and the map
//! files, plus the reader API for the trainer (8.2). Layout of a dataset directory
//! (`docs/formats.md` §20):
//!
//! ```text
//! manifest.json          format, code commit, config (+hash), source demos, maps, chunks, counts
//! players.json           anonymous player table: skill signals, rank, bucket (no names)
//! report.json            the aggregate report (reconstruction quality, techniques, ...)
//! maps/<sha256>.map      the maps the samples refer to (raw datafiles, local only)
//! chunks/NNNNN.pf.zst    zstd(postcard(Chunk)), one demo per chunk, <= chunk_frames frames
//! ```
//!
//! Writing is deterministic: chunks are cut by frame count, encoded single-threaded with a fixed
//! zstd level, JSON is written with a fixed field order, and every list is sorted by content
//! hashes, never by path or by completion order.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ddai_brain::{CharacterObservation, Observation};
use ddai_physics::map::MapData;
use ddai_physics::tuning::TuningParams;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{Config, hex};
use crate::skill::RankRow;
use crate::types::{Chunk, FORMAT_VERSION, FrameRec, ReplayClass, SampleRec, SkillBucket};

pub const FORMAT_NAME: &str = "ddai-human-dataset";
pub const MANIFEST: &str = "manifest.json";
pub const PLAYERS: &str = "players.json";
pub const REPORT: &str = "report.json";

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("i/o error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed json in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("chunk {path} cannot be decoded: {msg}")]
    Chunk { path: PathBuf, msg: String },
    #[error("sha256 mismatch for {path}: manifest says {expected}, file has {actual}")]
    Checksum {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("unsupported dataset format {0:?} version {1}")]
    Format(String, u32),
    #[error("map {0} is missing or unreadable")]
    Map(String),
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> DatasetError + '_ {
    move |source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// One source demo. Identified by its sha256 only - file names can contain nicknames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DemoEntry {
    pub sha256: String,
    pub size: u64,
    /// `"ok"` or `"skipped"`.
    pub status: String,
    /// Why the demo was skipped (generic text, never a path or a name).
    pub reason: Option<String>,
    pub map_name: Option<String>,
    /// Index into [`Manifest::maps`] of the map used (`None` for a skipped demo).
    pub map: Option<u32>,
    /// `"embedded"`, `"cache"` (found in the local map cache by name and crc) or `"none"`.
    pub map_source: String,
    pub frames: u64,
    pub samples: u64,
    /// The demo ended early on a decode error (frames before it were kept).
    pub decode_error: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapEntry {
    pub sha256: String,
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Path relative to the dataset directory.
    pub file: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkEntry {
    pub file: String,
    pub sha256: String,
    pub demo: u32,
    pub frames: u32,
    pub samples: u32,
    pub bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    pub demos_total: u64,
    pub demos_ok: u64,
    pub demos_skipped: u64,
    pub frames: u64,
    pub samples: u64,
    pub confident_samples: u64,
    pub players: u64,
    pub chunks: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub format_version: u32,
    pub name: String,
    /// Where the demos come from (free text without names, e.g. archive path and git commit).
    pub source: String,
    /// `git rev-parse HEAD` of the code that produced the dataset, or `"unknown"`.
    pub code_commit: String,
    pub config_hash: String,
    pub config: Config,
    pub demos: Vec<DemoEntry>,
    pub maps: Vec<MapEntry>,
    pub chunks: Vec<ChunkEntry>,
    pub counts: Counts,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Streams chunks into a dataset directory.
pub struct DatasetWriter {
    dir: PathBuf,
    level: i32,
    chunk_frames: usize,
    pub chunks: Vec<ChunkEntry>,
}

impl DatasetWriter {
    /// Creates `dir/chunks` and `dir/maps` (the directory itself may exist and be empty).
    pub fn create(dir: &Path, cfg: &Config) -> Result<Self, DatasetError> {
        fs::create_dir_all(dir.join("chunks")).map_err(io(dir))?;
        fs::create_dir_all(dir.join("maps")).map_err(io(dir))?;
        Ok(DatasetWriter {
            dir: dir.to_path_buf(),
            level: cfg.zstd_level,
            chunk_frames: cfg.chunk_frames.max(1),
            chunks: Vec::new(),
        })
    }

    /// Splits one demo into chunks at frame boundaries and writes them.
    pub fn write_demo(&mut self, demo: u32, frames: &[FrameRec], samples: &[SampleRec]) -> Result<(), DatasetError> {
        let mut start = 0usize;
        let mut sample_lo = 0usize;
        while start < frames.len() {
            let end = (start + self.chunk_frames).min(frames.len());
            let sample_hi = sample_lo + samples[sample_lo..].partition_point(|s| (s.frame as usize) < end);
            let chunk = Chunk {
                format_version: FORMAT_VERSION,
                demo,
                frames: frames[start..end].to_vec(),
                samples: samples[sample_lo..sample_hi]
                    .iter()
                    .map(|s| SampleRec {
                        frame: s.frame - start as u32,
                        ..*s
                    })
                    .collect(),
            };
            self.write_chunk(&chunk)?;
            start = end;
            sample_lo = sample_hi;
        }
        Ok(())
    }

    fn write_chunk(&mut self, chunk: &Chunk) -> Result<(), DatasetError> {
        let raw = postcard::to_stdvec(chunk).map_err(|e| DatasetError::Chunk {
            path: self.dir.clone(),
            msg: e.to_string(),
        })?;
        let packed = zstd::encode_all(raw.as_slice(), self.level).map_err(io(&self.dir))?;
        let file = format!("chunks/{:05}.pf.zst", self.chunks.len());
        let path = self.dir.join(&file);
        fs::write(&path, &packed).map_err(io(&path))?;
        self.chunks.push(ChunkEntry {
            file,
            sha256: sha256_hex(&packed),
            demo: chunk.demo,
            frames: chunk.frames.len() as u32,
            samples: chunk.samples.len() as u32,
            bytes: packed.len() as u64,
        });
        Ok(())
    }
}

/// Writes pretty JSON with a trailing newline.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), DatasetError> {
    let mut text = serde_json::to_string_pretty(value).map_err(|source| DatasetError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    text.push('\n');
    let mut f = fs::File::create(path).map_err(io(path))?;
    f.write_all(text.as_bytes()).map_err(io(path))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, DatasetError> {
    let text = fs::read(path).map_err(io(path))?;
    serde_json::from_slice(&text).map_err(|source| DatasetError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Which samples a reader yields. The default yields everything.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    /// Keep samples with at least one of these tag bits (0 = no requirement).
    pub any_tags: u32,
    /// Keep samples with all of these tag bits.
    pub all_tags: u32,
    /// Drop samples with any of these tag bits.
    pub none_tags: u32,
    /// Minimum skill bucket of the acting player (`Unranked` < `Low` < `Mid` < `Top`).
    pub min_skill: SkillBucket,
    /// Minimum replay class; `Within1px` keeps the confidently reconstructed samples.
    pub min_replay: ReplayClass,
    /// Only samples whose next wire core was fresh (the inputs were observable).
    pub require_next_fresh: bool,
    /// Only samples where something happens (non-neutral action, movement or hooking).
    pub require_active: bool,
    /// Drop samples where the acting character is frozen (its input is ignored by the server).
    pub exclude_frozen: bool,
    /// Restrict to these maps (sha256 hex); empty = all.
    pub maps: Vec<String>,
    /// Restrict to these demos (indices in [`Manifest::demos`]); empty = all. With `skip_demos`
    /// this gives a leak-free train/validation split by demo.
    pub demos: Vec<u32>,
    /// Drop these demos.
    pub skip_demos: Vec<u32>,
}

impl Default for Filter {
    fn default() -> Self {
        Filter {
            any_tags: 0,
            all_tags: 0,
            none_tags: 0,
            min_skill: SkillBucket::Unranked,
            min_replay: ReplayClass::Unavailable,
            require_next_fresh: false,
            require_active: false,
            exclude_frozen: false,
            maps: Vec::new(),
            demos: Vec::new(),
            skip_demos: Vec::new(),
        }
    }
}

impl Filter {
    /// The trainer's default: confidently reconstructed, not frozen, players of at least `Mid` skill.
    pub fn good_play() -> Self {
        Filter {
            min_skill: SkillBucket::Mid,
            min_replay: ReplayClass::Within1px,
            exclude_frozen: true,
            ..Filter::default()
        }
    }

    /// Whether samples of demo `demo` (on map `map_sha256`) can pass this filter at all.
    pub fn keeps_demo(&self, demo: u32, map_sha256: Option<&str>) -> bool {
        if (!self.demos.is_empty() && !self.demos.contains(&demo)) || self.skip_demos.contains(&demo) {
            return false;
        }
        self.maps.is_empty() || map_sha256.is_some_and(|m| self.maps.iter().any(|x| x == m))
    }

    fn keeps(&self, s: &SampleRec, frame: &FrameRec) -> bool {
        if s.tags & self.none_tags != 0 || s.tags & self.all_tags != self.all_tags {
            return false;
        }
        if self.any_tags != 0 && s.tags & self.any_tags == 0 {
            return false;
        }
        if SkillBucket::from_u8(s.skill) < self.min_skill || s.replay() < self.min_replay {
            return false;
        }
        if self.require_next_fresh && !s.next_fresh() || self.require_active && !s.active() {
            return false;
        }
        if self.exclude_frozen && frame.chars.get(s.slot as usize).is_some_and(|c| c.frozen()) {
            return false;
        }
        true
    }
}

/// Identity and labels of a sample.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleMeta {
    pub map_sha256: [u8; 32],
    pub demo_sha256: [u8; 32],
    /// Index of the demo in [`Manifest::demos`].
    pub demo: u32,
    pub tick: i32,
    /// Anonymous player label, unique per `(client id, stint)` inside the demo.
    pub player: u16,
    pub tags: u32,
    pub skill: SkillBucket,
}

/// Reconstruction quality of a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleQuality {
    pub replay: ReplayClass,
    pub next_fresh: bool,
    pub active: bool,
    /// `replay >= Within1px`: the sample deserves training weight.
    pub confident: bool,
}

/// One training sample.
#[derive(Debug, Clone)]
pub struct Sample {
    pub meta: SampleMeta,
    pub observation: Observation,
    pub action: ddai_brain::Action,
    pub quality: SampleQuality,
}

fn parse_sha(hexs: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = hexs
            .get(2 * i..2 * i + 2)
            .and_then(|h| u8::from_str_radix(h, 16).ok())
            .unwrap_or(0);
    }
    out
}

/// Reader for the trainer: manifest, players, and lazily decoded chunks.
pub struct DatasetReader {
    dir: PathBuf,
    pub manifest: Manifest,
    pub players: Vec<RankRow>,
    maps: std::sync::Mutex<HashMap<u32, Arc<MapData>>>,
}

impl DatasetReader {
    pub fn open(dir: &Path) -> Result<Self, DatasetError> {
        let manifest: Manifest = read_json(&dir.join(MANIFEST))?;
        if manifest.format != FORMAT_NAME || manifest.format_version != FORMAT_VERSION {
            return Err(DatasetError::Format(manifest.format, manifest.format_version));
        }
        let players: Vec<RankRow> = read_json(&dir.join(PLAYERS))?;
        Ok(DatasetReader {
            dir: dir.to_path_buf(),
            manifest,
            players,
            maps: Default::default(),
        })
    }

    /// Checks every chunk file against the sha256 in the manifest.
    pub fn verify(&self) -> Result<(), DatasetError> {
        for c in &self.manifest.chunks {
            let path = self.dir.join(&c.file);
            let bytes = fs::read(&path).map_err(io(&path))?;
            let actual = sha256_hex(&bytes);
            if actual != c.sha256 {
                return Err(DatasetError::Checksum {
                    path,
                    expected: c.sha256.clone(),
                    actual,
                });
            }
        }
        Ok(())
    }

    fn map(&self, idx: u32) -> Result<Arc<MapData>, DatasetError> {
        if let Some(m) = self.maps.lock().expect("map cache lock").get(&idx) {
            return Ok(Arc::clone(m));
        }
        let entry = self
            .manifest
            .maps
            .get(idx as usize)
            .ok_or_else(|| DatasetError::Map(format!("#{idx}")))?;
        let path = self.dir.join(&entry.file);
        let bytes = fs::read(&path).map_err(io(&path))?;
        let loaded = ddai_map::load_map(&bytes).map_err(|_| DatasetError::Map(entry.sha256.clone()))?;
        let arc = Arc::new(loaded.data);
        self.maps.lock().expect("map cache lock").insert(idx, Arc::clone(&arc));
        Ok(arc)
    }

    /// Decodes one chunk (by index in [`Manifest::chunks`]).
    pub fn read_chunk(&self, index: usize) -> Result<Chunk, DatasetError> {
        let entry = self.manifest.chunks.get(index).ok_or_else(|| DatasetError::Chunk {
            path: self.dir.clone(),
            msg: format!("no chunk {index}"),
        })?;
        let path = self.dir.join(&entry.file);
        let bytes = fs::read(&path).map_err(io(&path))?;
        let raw = zstd::decode_all(bytes.as_slice()).map_err(|e| DatasetError::Chunk {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        postcard::from_bytes(&raw).map_err(|e| DatasetError::Chunk {
            path,
            msg: e.to_string(),
        })
    }

    /// Indices of the chunks that can contain samples passing `filter`'s demo and map selection
    /// (chunks of demos without a map are never returned).
    pub fn chunks_for(&self, filter: &Filter) -> Vec<usize> {
        let m = &self.manifest;
        (0..m.chunks.len())
            .filter(|&ci| {
                let demo = m.chunks[ci].demo;
                let Some(d) = m.demos.get(demo as usize) else {
                    return false;
                };
                let Some(map_idx) = d.map else { return false };
                filter.keeps_demo(demo, m.maps.get(map_idx as usize).map(|e| e.sha256.as_str()))
            })
            .collect()
    }

    /// The samples of chunk `index` (an index into [`Manifest::chunks`]) that pass `filter`, in
    /// dataset order. Independent per chunk and callable from several threads at once (the reader
    /// is `Sync`), so a trainer can shuffle the chunk order and decode chunks in parallel.
    pub fn samples_in_chunk(&self, index: usize, filter: &Filter) -> Result<Vec<Sample>, DatasetError> {
        let m = &self.manifest;
        let entry = m.chunks.get(index).ok_or_else(|| DatasetError::Chunk {
            path: self.dir.clone(),
            msg: format!("no chunk {index}"),
        })?;
        let demo = &m.demos[entry.demo as usize];
        let Some(map_idx) = demo.map else { return Ok(Vec::new()) };
        let map_entry = &m.maps[map_idx as usize];
        if !filter.keeps_demo(entry.demo, Some(&map_entry.sha256)) {
            return Ok(Vec::new());
        }
        let map = self.map(map_idx)?;
        let chunk = self.read_chunk(index)?;
        let (map_sha, demo_sha) = (parse_sha(&map_entry.sha256), parse_sha(&demo.sha256));
        let mut out = Vec::new();
        for s in &chunk.samples {
            let frame = &chunk.frames[s.frame as usize];
            if !filter.keeps(s, frame) {
                continue;
            }
            let me = &frame.chars[s.slot as usize];
            let others: Vec<CharacterObservation> = frame
                .chars
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != s.slot as usize)
                .map(|(_, c)| c.to_observation())
                .collect();
            out.push(Sample {
                meta: SampleMeta {
                    map_sha256: map_sha,
                    demo_sha256: demo_sha,
                    demo: chunk.demo,
                    tick: frame.tick,
                    player: me.player,
                    tags: s.tags,
                    skill: SkillBucket::from_u8(s.skill),
                },
                observation: Observation {
                    map: Arc::clone(&map),
                    tick: frame.tick,
                    self_state: me.to_observation(),
                    others,
                    target_id: (s.target >= 0).then_some(i32::from(s.target)),
                    tuning: TuningParams::default(),
                },
                action: s.action.to_action(),
                quality: SampleQuality {
                    replay: s.replay(),
                    next_fresh: s.next_fresh(),
                    active: s.active(),
                    confident: s.confident(),
                },
            });
        }
        Ok(out)
    }

    /// Iterates the samples that pass `filter`, chunk by chunk (one chunk in memory at a time), in
    /// dataset order.
    pub fn samples<'a>(&'a self, filter: &'a Filter) -> SampleIter<'a> {
        SampleIter {
            reader: self,
            filter,
            chunks: self.chunks_for(filter).into_iter(),
            current: Vec::new().into_iter(),
            failed: false,
        }
    }
}

pub struct SampleIter<'a> {
    reader: &'a DatasetReader,
    filter: &'a Filter,
    chunks: std::vec::IntoIter<usize>,
    current: std::vec::IntoIter<Sample>,
    failed: bool,
}

impl Iterator for SampleIter<'_> {
    type Item = Result<Sample, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            if let Some(s) = self.current.next() {
                return Some(Ok(s));
            }
            let ci = self.chunks.next()?;
            match self.reader.samples_in_chunk(ci, self.filter) {
                Ok(v) => self.current = v.into_iter(),
                Err(e) => {
                    self.failed = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

#[cfg(all(test, feature = "test-util"))]
mod tests {
    use super::*;
    use crate::types::{ActionRec, CharRec, char_flags};

    fn char_rec(id: u8, player: u16, x: f32) -> CharRec {
        CharRec {
            id,
            player,
            team: 0,
            pos: [x, 100.0],
            vel: [0.5, -1.5],
            hook_state: 0,
            hook_pos: [x, 100.0],
            hooked_player: -1,
            flags: char_flags::FRESH,
            freeze_ticks: 0,
            jumps_left: 2,
            jumps_used: 0,
            weapon: 0,
            direction: 1,
            aim: [1, 0],
        }
    }

    fn sample(frame: u32, slot: u8, tags: u32, replay: ReplayClass, skill: SkillBucket) -> SampleRec {
        SampleRec {
            frame,
            slot,
            action: ActionRec {
                direction: 1,
                jump: false,
                hook: true,
                fire: false,
                aim: [3, -4],
            },
            target: 1,
            tags,
            q: replay as u8,
            skill: skill as u8,
        }
    }

    /// 8 frames of two characters, samples for character 0 on most frames.
    fn demo_data() -> (Vec<FrameRec>, Vec<SampleRec>) {
        let frames: Vec<FrameRec> = (0..8)
            .map(|i| FrameRec {
                tick: 100 + 2 * i,
                chars: vec![char_rec(0, 7, i as f32), char_rec(1, 9, 50.0 + i as f32)],
            })
            .collect();
        let samples = vec![
            sample(0, 0, 0, ReplayClass::Exact, SkillBucket::Top),
            sample(1, 0, 0x100, ReplayClass::Off, SkillBucket::Top),
            sample(2, 1, 0, ReplayClass::Within1px, SkillBucket::Low),
            sample(4, 0, 0x100, ReplayClass::Exact, SkillBucket::Mid),
            sample(5, 1, 0x4, ReplayClass::Exact, SkillBucket::Unranked),
            sample(7, 0, 0, ReplayClass::Exact, SkillBucket::Top),
        ];
        (frames, samples)
    }

    fn write_fixture(dir: &Path) -> Manifest {
        let cfg = Config {
            chunk_frames: 3,
            ..Config::default()
        };
        let map = crate::synth::map_bytes(6, 6, (2, 4));
        let map_sha = sha256_hex(&map);
        fs::create_dir_all(dir.join("maps")).unwrap();
        fs::write(dir.join(format!("maps/{map_sha}.map")), &map).unwrap();
        let mut w = DatasetWriter::create(dir, &cfg).unwrap();
        let (frames, samples) = demo_data();
        w.write_demo(0, &frames, &samples).unwrap();
        let m = Manifest {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            name: "t".to_string(),
            source: "synthetic".to_string(),
            code_commit: "c".to_string(),
            config_hash: cfg.hash_hex(),
            config: cfg,
            demos: vec![DemoEntry {
                sha256: "ab".repeat(32),
                size: 1,
                status: "ok".to_string(),
                reason: None,
                map_name: Some("m".to_string()),
                map: Some(0),
                map_source: "embedded".to_string(),
                frames: 8,
                samples: 6,
                decode_error: false,
            }],
            maps: vec![MapEntry {
                sha256: map_sha.clone(),
                name: "m".to_string(),
                width: 6,
                height: 6,
                file: format!("maps/{map_sha}.map"),
            }],
            chunks: w.chunks.clone(),
            counts: Counts::default(),
        };
        write_json(&dir.join(MANIFEST), &m).unwrap();
        write_json(&dir.join(PLAYERS), &Vec::<RankRow>::new()).unwrap();
        m
    }

    #[test]
    fn a_demo_is_split_at_frame_boundaries_and_read_back_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let m = write_fixture(tmp.path());
        assert_eq!(m.chunks.len(), 3, "8 frames in chunks of 3");
        assert_eq!(m.chunks.iter().map(|c| c.frames).collect::<Vec<_>>(), vec![3, 3, 2]);
        assert_eq!(m.chunks.iter().map(|c| c.samples).collect::<Vec<_>>(), vec![3, 2, 1]);
        let r = DatasetReader::open(tmp.path()).unwrap();
        r.verify().unwrap();
        let all: Vec<Sample> = r.samples(&Filter::default()).collect::<Result<_, _>>().unwrap();
        let ticks: Vec<i32> = all.iter().map(|s| s.meta.tick).collect();
        assert_eq!(
            ticks,
            vec![100, 102, 104, 108, 110, 114],
            "chunk-relative frame indices are rebased"
        );
        // The acting character is the slot, the rest are `others`, the target is a client id.
        let s = &all[2];
        assert_eq!(s.observation.self_state.id, 1);
        assert_eq!(s.meta.player, 9);
        assert_eq!(s.observation.others.len(), 1);
        assert_eq!(s.observation.others[0].id, 0);
        assert_eq!(s.observation.target_id, Some(1));
        assert_eq!(s.observation.tick, 104);
        assert_eq!(s.observation.self_state.pos.x, 50.0 + 2.0);
        assert_eq!(s.action.target, ddai_brain::IVec2::new(3, -4));
        assert!(s.action.hook && !s.action.jump);
        assert_eq!(s.meta.skill, SkillBucket::Low);
        assert_eq!(s.observation.map.width, 6);
        assert!(
            all.iter()
                .all(|s| Arc::ptr_eq(&s.observation.map, &all[0].observation.map)),
            "one shared map"
        );
    }

    #[test]
    fn filters_select_by_tags_skill_and_confidence() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path());
        let r = DatasetReader::open(tmp.path()).unwrap();
        let ticks = |f: &Filter| -> Vec<i32> { r.samples(f).map(|s| s.unwrap().meta.tick).collect() };
        assert_eq!(
            ticks(&Filter {
                any_tags: 0x100,
                ..Filter::default()
            }),
            vec![102, 108]
        );
        assert_eq!(
            ticks(&Filter {
                any_tags: 0x100 | 0x4,
                ..Filter::default()
            }),
            vec![102, 108, 110]
        );
        assert_eq!(
            ticks(&Filter {
                all_tags: 0x100,
                none_tags: 0x4,
                ..Filter::default()
            }),
            vec![102, 108]
        );
        assert_eq!(
            ticks(&Filter {
                none_tags: 0x100,
                ..Filter::default()
            }),
            vec![100, 104, 110, 114]
        );
        assert_eq!(
            ticks(&Filter {
                min_skill: SkillBucket::Mid,
                ..Filter::default()
            }),
            vec![100, 102, 108, 114]
        );
        assert_eq!(
            ticks(&Filter {
                min_skill: SkillBucket::Top,
                ..Filter::default()
            }),
            vec![100, 102, 114]
        );
        assert_eq!(
            ticks(&Filter {
                min_replay: ReplayClass::Within1px,
                ..Filter::default()
            }),
            vec![100, 104, 108, 110, 114]
        );
        assert_eq!(
            ticks(&Filter {
                min_replay: ReplayClass::Exact,
                ..Filter::default()
            }),
            vec![100, 108, 110, 114]
        );
        assert_eq!(ticks(&Filter::good_play()), vec![100, 108, 114]);
        // Combining everything empties the set without error.
        assert!(
            ticks(&Filter {
                any_tags: 0x4,
                min_skill: SkillBucket::Top,
                ..Filter::default()
            })
            .is_empty()
        );
        // Map filter.
        let sha = r.manifest.maps[0].sha256.clone();
        assert_eq!(
            ticks(&Filter {
                maps: vec![sha],
                ..Filter::default()
            })
            .len(),
            6
        );
        assert!(
            ticks(&Filter {
                maps: vec!["f".repeat(64)],
                ..Filter::default()
            })
            .is_empty()
        );
    }

    #[test]
    fn exclude_frozen_and_flags_use_the_frame_state() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = Config::default();
        let map = crate::synth::map_bytes(6, 6, (2, 4));
        let map_sha = sha256_hex(&map);
        fs::create_dir_all(tmp.path().join("maps")).unwrap();
        fs::write(tmp.path().join(format!("maps/{map_sha}.map")), &map).unwrap();
        let mut w = DatasetWriter::create(tmp.path(), &cfg).unwrap();
        let mut frames: Vec<FrameRec> = (0..2)
            .map(|i| FrameRec {
                tick: 10 + 2 * i,
                chars: vec![char_rec(0, 1, 0.0)],
            })
            .collect();
        frames[1].chars[0].flags |= char_flags::FROZEN;
        let mut s0 = sample(0, 0, 0, ReplayClass::Exact, SkillBucket::Low);
        s0.q |= crate::types::quality::ACTIVE | crate::types::quality::NEXT_FRESH;
        let s1 = sample(1, 0, 0, ReplayClass::Exact, SkillBucket::Low);
        w.write_demo(0, &frames, &[s0, s1]).unwrap();
        let mut m = Manifest {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            name: "t".into(),
            source: "s".into(),
            code_commit: "c".into(),
            config_hash: cfg.hash_hex(),
            config: cfg,
            demos: vec![],
            maps: vec![MapEntry {
                sha256: map_sha.clone(),
                name: "m".into(),
                width: 6,
                height: 6,
                file: format!("maps/{map_sha}.map"),
            }],
            chunks: w.chunks.clone(),
            counts: Counts::default(),
        };
        m.demos.push(DemoEntry {
            sha256: "cd".repeat(32),
            size: 1,
            status: "ok".into(),
            reason: None,
            map_name: None,
            map: Some(0),
            map_source: "embedded".into(),
            frames: 2,
            samples: 2,
            decode_error: false,
        });
        write_json(&tmp.path().join(MANIFEST), &m).unwrap();
        write_json(&tmp.path().join(PLAYERS), &Vec::<RankRow>::new()).unwrap();
        let r = DatasetReader::open(tmp.path()).unwrap();
        let n = |f: &Filter| r.samples(f).count();
        assert_eq!(n(&Filter::default()), 2);
        assert_eq!(
            n(&Filter {
                exclude_frozen: true,
                ..Filter::default()
            }),
            1
        );
        assert_eq!(
            n(&Filter {
                require_active: true,
                ..Filter::default()
            }),
            1
        );
        assert_eq!(
            n(&Filter {
                require_next_fresh: true,
                ..Filter::default()
            }),
            1
        );
        let first = r.samples(&Filter::default()).next().unwrap().unwrap();
        assert!(first.quality.confident && first.quality.active && first.quality.next_fresh);
        assert!(!r.samples(&Filter::default()).nth(1).unwrap().unwrap().quality.active);
        assert!(
            r.samples(&Filter::default())
                .nth(1)
                .unwrap()
                .unwrap()
                .observation
                .self_state
                .is_frozen
        );
    }

    fn two_demo_fixture(dir: &Path) {
        let cfg = Config {
            chunk_frames: 3,
            ..Config::default()
        };
        let map = crate::synth::map_bytes(6, 6, (2, 4));
        let map_sha = sha256_hex(&map);
        fs::create_dir_all(dir.join("maps")).unwrap();
        fs::write(dir.join(format!("maps/{map_sha}.map")), &map).unwrap();
        let mut w = DatasetWriter::create(dir, &cfg).unwrap();
        let (frames, samples) = demo_data();
        w.write_demo(0, &frames, &samples).unwrap();
        let later: Vec<FrameRec> = frames
            .iter()
            .map(|f| FrameRec {
                tick: f.tick + 1000,
                chars: f.chars.clone(),
            })
            .collect();
        w.write_demo(1, &later, &samples).unwrap();
        let demo = |sha: &str| DemoEntry {
            sha256: sha.repeat(32),
            size: 1,
            status: "ok".into(),
            reason: None,
            map_name: None,
            map: Some(0),
            map_source: "embedded".into(),
            frames: 8,
            samples: 6,
            decode_error: false,
        };
        let m = Manifest {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            name: "t".into(),
            source: "s".into(),
            code_commit: "c".into(),
            config_hash: cfg.hash_hex(),
            config: cfg,
            demos: vec![demo("ab"), demo("cd")],
            maps: vec![MapEntry {
                sha256: map_sha.clone(),
                name: "m".into(),
                width: 6,
                height: 6,
                file: format!("maps/{map_sha}.map"),
            }],
            chunks: w.chunks.clone(),
            counts: Counts::default(),
        };
        write_json(&dir.join(MANIFEST), &m).unwrap();
        write_json(&dir.join(PLAYERS), &Vec::<RankRow>::new()).unwrap();
    }

    #[test]
    fn demo_selection_gives_a_leak_free_split_and_chunks_can_be_read_in_parallel() {
        fn assert_sync<T: Sync + Send>() {}
        assert_sync::<DatasetReader>();
        let tmp = tempfile::tempdir().unwrap();
        two_demo_fixture(tmp.path());
        let r = DatasetReader::open(tmp.path()).unwrap();
        let all: Vec<Sample> = r.samples(&Filter::default()).collect::<Result<_, _>>().unwrap();
        assert_eq!(all.len(), 12);
        let val = Filter {
            demos: vec![1],
            ..Filter::default()
        };
        let train = Filter {
            skip_demos: vec![1],
            ..Filter::default()
        };
        let v: Vec<Sample> = r.samples(&val).collect::<Result<_, _>>().unwrap();
        let t: Vec<Sample> = r.samples(&train).collect::<Result<_, _>>().unwrap();
        assert_eq!((v.len(), t.len()), (6, 6));
        assert!(v.iter().all(|s| s.meta.demo == 1 && s.meta.tick >= 1000));
        assert!(t.iter().all(|s| s.meta.demo == 0 && s.meta.tick < 1000));
        // Selecting nothing valid yields nothing; demo and other filters compose.
        assert_eq!(
            r.samples(&Filter {
                demos: vec![7],
                ..Filter::default()
            })
            .count(),
            0
        );
        assert_eq!(
            r.samples(&Filter {
                demos: vec![1],
                any_tags: 0x100,
                ..Filter::default()
            })
            .count(),
            2
        );
        // Chunk-level API: chunks of the split are disjoint and cover the dataset.
        let cv = r.chunks_for(&val);
        let ct = r.chunks_for(&train);
        assert_eq!(cv.len(), 3);
        assert_eq!(ct.len(), 3);
        assert!(cv.iter().all(|c| !ct.contains(c)));
        assert!(
            r.samples_in_chunk(ct[0], &val).unwrap().is_empty(),
            "a chunk of another demo yields nothing"
        );
        // Reading every chunk from several threads gives the sequential result.
        let f = Filter::default();
        let order: Vec<usize> = r.chunks_for(&f);
        let parallel: Vec<Vec<i32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = order
                .iter()
                .map(|&ci| {
                    let (r, f) = (&r, &f);
                    scope.spawn(move || {
                        r.samples_in_chunk(ci, f)
                            .unwrap()
                            .iter()
                            .map(|s| s.meta.tick)
                            .collect::<Vec<i32>>()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let flat: Vec<i32> = parallel.into_iter().flatten().collect();
        assert_eq!(flat, all.iter().map(|s| s.meta.tick).collect::<Vec<_>>());
        assert!(r.samples_in_chunk(999, &f).is_err());
    }

    #[test]
    fn a_wrong_format_version_or_name_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let mut m = write_fixture(tmp.path());
        m.format_version += 1;
        write_json(&tmp.path().join(MANIFEST), &m).unwrap();
        assert!(matches!(
            DatasetReader::open(tmp.path()),
            Err(DatasetError::Format(_, _))
        ));
        m.format_version = FORMAT_VERSION;
        m.format = "something-else".to_string();
        write_json(&tmp.path().join(MANIFEST), &m).unwrap();
        assert!(matches!(
            DatasetReader::open(tmp.path()),
            Err(DatasetError::Format(_, _))
        ));
    }

    #[test]
    fn sha_parsing_round_trips_and_tolerates_garbage() {
        let s = sha256_hex(b"x");
        assert_eq!(config::hex(&parse_sha(&s)), s);
        assert_eq!(parse_sha("zz"), [0u8; 32]);
    }

    use crate::config;
}
