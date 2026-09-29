//! `dataset from-demos`: discover demos, resolve maps, process every demo on a small thread pool,
//! rank the players, write the dataset and the report - deterministically (results are collected
//! by index, the demo order is the sha256 order, nothing depends on paths or thread timing).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ddai_demo::Demo;

use crate::config::Config;
use crate::dataset::{
    self, Counts, DatasetError, DatasetWriter, DemoEntry, MANIFEST, Manifest, MapEntry, PLAYERS, REPORT, sha256_hex,
    write_json,
};
use crate::demo::{self, DemoOutput};
use crate::ingest;
use crate::report::{self, DemoCounts, Report};
use crate::skill::RankRow;
use crate::types::{FORMAT_VERSION, SkillBucket};

/// Demos larger than this are skipped (the reader's own limit).
pub const MAX_DEMO_BYTES: u64 = ddai_demo::header::MAX_DEMO_FILE_SIZE as u64;

#[derive(Debug, Clone)]
pub struct Options {
    pub demos_dir: PathBuf,
    pub out_dir: PathBuf,
    /// Directories searched (recursively) for `.map` files to use when a demo has no usable
    /// embedded map: matched by the demo header's crc and size.
    pub map_dirs: Vec<PathBuf>,
    pub threads: usize,
    pub code_commit: String,
    pub name: String,
    pub source: String,
    /// Process only the first `n` demos (in sha256 order); for smoke tests.
    pub limit: Option<usize>,
    pub top_players: usize,
}

/// Recursively lists `*.demo` files, sorted by path (the order is only used to find files; the
/// dataset order is by sha256).
pub fn discover(dir: &Path, ext: &str) -> Result<Vec<PathBuf>, DatasetError> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rd = fs::read_dir(&d).map_err(|source| DatasetError::Io {
            path: d.clone(),
            source,
        })?;
        for e in rd {
            let e = e.map_err(|source| DatasetError::Io {
                path: d.clone(),
                source,
            })?;
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case(ext)) {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// A map file from the local cache.
struct CachedMap {
    crc: u32,
    size: usize,
    file_stem: String,
    bytes: Vec<u8>,
}

fn load_map_cache(dirs: &[PathBuf]) -> Vec<CachedMap> {
    let mut out = Vec::new();
    for d in dirs {
        let Ok(files) = discover(d, "map") else { continue };
        for f in files {
            let Ok(bytes) = fs::read(&f) else { continue };
            let Ok(loaded) = ddai_map::load_map(&bytes) else {
                continue;
            };
            out.push(CachedMap {
                crc: loaded.crc32,
                size: loaded.size,
                file_stem: f
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                bytes,
            });
        }
    }
    out.sort_by(|a, b| (a.crc, &a.file_stem).cmp(&(b.crc, &b.file_stem)));
    out
}

/// The map of a demo: the embedded one if `ddai-map` can load it, else a cached file whose crc
/// and size equal the demo header's (a file name starting with the map name wins ties).
fn resolve_map(
    demo: &Demo<'_>,
    cache: &[CachedMap],
) -> Result<(Vec<u8>, ddai_physics::map::MapData, &'static str), &'static str> {
    let embedded = demo.map_bytes();
    let mut embedded_failed = false;
    if !embedded.is_empty() {
        match ddai_map::load_map(embedded) {
            Ok(m) => return Ok((embedded.to_vec(), m.data, "embedded")),
            Err(_) => embedded_failed = true,
        }
    }
    let h = &demo.header;
    let mut cands: Vec<&CachedMap> = cache
        .iter()
        .filter(|c| c.crc == h.map_crc && (h.map_size == 0 || c.size == h.map_size as usize))
        .collect();
    cands.sort_by_key(|c| !c.file_stem.starts_with(&h.map_name));
    if let Some(c) = cands.first()
        && let Ok(m) = ddai_map::load_map(&c.bytes)
    {
        return Ok((c.bytes.clone(), m.data, "cache"));
    }
    Err(if embedded_failed {
        "embedded map unreadable and no cached map with the demo's crc"
    } else {
        "no embedded map and no cached map with the demo's crc"
    })
}

struct Processed {
    entry: DemoEntry,
    out: Option<DemoOutput>,
    /// sha256 of the map the demo was processed with.
    map_sha: Option<String>,
}

fn skipped(sha: String, size: u64, reason: &str) -> Processed {
    Processed {
        entry: DemoEntry {
            sha256: sha,
            size,
            status: "skipped".to_string(),
            reason: Some(reason.to_string()),
            map_name: None,
            map: None,
            map_source: "none".to_string(),
            frames: 0,
            samples: 0,
            decode_error: false,
        },
        out: None,
        map_sha: None,
    }
}

/// Builds the dataset. Returns the report (also written to `report.json`).
pub fn from_demos(opts: &Options, cfg: &Config, log: &(dyn Fn(&str) + Sync)) -> Result<Report, DatasetError> {
    let files = discover(&opts.demos_dir, "demo")?;
    // Hash everything first: the dataset order is by sha256.
    let mut items: Vec<(String, PathBuf, u64)> = Vec::new();
    let mut counts = DemoCounts::default();
    let mut seen = std::collections::BTreeSet::new();
    for f in files {
        let size = fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        if size > MAX_DEMO_BYTES {
            counts.total += 1;
            counts.skipped += 1;
            *counts
                .skipped_reasons
                .entry("file larger than the reader limit".into())
                .or_default() += 1;
            continue;
        }
        let bytes = fs::read(&f).map_err(|source| DatasetError::Io {
            path: f.clone(),
            source,
        })?;
        let sha = sha256_hex(&bytes);
        counts.total += 1;
        if !seen.insert(sha.clone()) {
            counts.duplicates += 1;
            continue;
        }
        items.push((sha, f, size));
    }
    items.sort();
    if let Some(n) = opts.limit {
        items.truncate(n);
        // A smoke run reports only the demos it looked at.
        counts.total = items.len() as u64;
        counts.duplicates = 0;
    }
    log(&format!(
        "{} demos to process ({} duplicates dropped)",
        items.len(),
        counts.duplicates
    ));

    let cache = load_map_cache(&opts.map_dirs);
    let maps_used: Mutex<BTreeMap<String, (String, Vec<u8>)>> = Mutex::new(BTreeMap::new());
    let results: Vec<Mutex<Option<Processed>>> = (0..items.len()).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let threads = opts.threads.clamp(1, 6);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= items.len() {
                        break;
                    }
                    let (sha, path, size) = &items[i];
                    let started = std::time::Instant::now();
                    let processed = process_one(cfg, sha, path, *size, &cache, &maps_used);
                    if started.elapsed().as_secs() >= 20 {
                        log(&format!(
                            "slow demo {}: {} frames, {:.0} s",
                            &sha[..12],
                            processed.entry.frames,
                            started.elapsed().as_secs_f32()
                        ));
                    }
                    *results[i].lock().expect("no panics while holding the lock") = Some(processed);
                    let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                    if n.is_multiple_of(20) || n == items.len() {
                        log(&format!("processed {n}/{}", items.len()));
                    }
                }
            });
        }
    });
    let processed: Vec<Processed> = results
        .into_iter()
        .map(|m| {
            m.into_inner()
                .expect("no poisoned lock")
                .expect("every index was processed")
        })
        .collect();

    // Map table (sorted by sha256) and demo -> map index.
    let maps_used = maps_used.into_inner().expect("no poisoned lock");
    let map_index: HashMap<String, u32> = maps_used
        .keys()
        .enumerate()
        .map(|(i, k)| (k.clone(), i as u32))
        .collect();
    let mut demos: Vec<DemoEntry> = Vec::new();
    let mut outputs: Vec<Option<DemoOutput>> = Vec::new();
    for p in processed {
        let mut e = p.entry;
        if e.status == "ok" {
            counts.ok += 1;
            if e.map_source == "embedded" {
                counts.map_embedded += 1;
            } else {
                counts.map_from_cache += 1;
            }
            if e.decode_error {
                counts.decode_errors += 1;
            }
            e.map = p.map_sha.as_ref().and_then(|s| map_index.get(s)).copied();
            if p.out.as_ref().is_some_and(|o| o.tune_nondefault_frames > 0) {
                counts.tune_nondefault += 1;
            }
            if p.out.as_ref().is_some_and(|o| o.tune_movement_frames > 0) {
                counts.tune_movement_nondefault += 1;
            }
        } else {
            counts.skipped += 1;
            *counts
                .skipped_reasons
                .entry(e.reason.clone().unwrap_or_default())
                .or_default() += 1;
        }
        demos.push(e);
        outputs.push(p.out);
    }
    finish(opts, cfg, counts, demos, outputs, maps_used, log)
}

fn process_one(
    cfg: &Config,
    sha: &str,
    path: &Path,
    size: u64,
    cache: &[CachedMap],
    maps_used: &Mutex<BTreeMap<String, (String, Vec<u8>)>>,
) -> Processed {
    let Ok(bytes) = fs::read(path) else {
        return skipped(sha.to_string(), size, "file unreadable");
    };
    let demo = match Demo::parse(&bytes) {
        Ok(d) => d,
        Err(_) => return skipped(sha.to_string(), size, "demo header unreadable"),
    };
    let (map_bytes, map, source) = match resolve_map(&demo, cache) {
        Ok(x) => x,
        Err(reason) => return skipped(sha.to_string(), size, reason),
    };
    let map_sha = sha256_hex(&map_bytes);
    maps_used
        .lock()
        .expect("no panics while holding the lock")
        .entry(map_sha.clone())
        .or_insert_with(|| (demo.header.map_name.clone(), map_bytes));
    let ing = ingest::ingest(&demo);
    if ing.frames.is_empty() {
        return skipped(sha.to_string(), size, "no snapshots");
    }
    let out = demo::process(cfg, &Arc::new(map), &ing);
    Processed {
        entry: DemoEntry {
            sha256: sha.to_string(),
            size,
            status: "ok".to_string(),
            reason: None,
            map_name: Some(demo.header.map_name.clone()),
            map: None,
            map_source: source.to_string(),
            frames: out.frames.len() as u64,
            samples: out.samples.len() as u64,
            decode_error: ing.decode_error.is_some(),
        },
        out: Some(out),
        map_sha: Some(map_sha),
    }
}

fn finish(
    opts: &Options,
    cfg: &Config,
    counts: DemoCounts,
    demos: Vec<DemoEntry>,
    mut outputs: Vec<Option<DemoOutput>>,
    maps_used: BTreeMap<String, (String, Vec<u8>)>,
    log: &(dyn Fn(&str) + Sync),
) -> Result<Report, DatasetError> {
    // Global ranking.
    let mut rows: Vec<RankRow> = Vec::new();
    for (di, out) in outputs.iter().enumerate() {
        if let Some(out) = out {
            for p in &out.players {
                rows.push(RankRow {
                    demo: di as u32,
                    skill: p.clone(),
                    bucket: SkillBucket::Unranked,
                    rank: 0,
                });
            }
        }
    }
    let ranking = crate::skill::assign_buckets(rows, cfg);
    let bucket_of: HashMap<(u32, u16), SkillBucket> =
        ranking.iter().map(|r| ((r.demo, r.skill.player), r.bucket)).collect();
    for (di, out) in outputs.iter_mut().enumerate() {
        if let Some(out) = out {
            for s in out.samples.iter_mut() {
                let label = out.frames[s.frame as usize].chars[s.slot as usize].player;
                s.skill = bucket_of
                    .get(&(di as u32, label))
                    .copied()
                    .unwrap_or(SkillBucket::Unranked) as u8;
            }
        }
    }

    // Write.
    fs::create_dir_all(&opts.out_dir).map_err(|source| DatasetError::Io {
        path: opts.out_dir.clone(),
        source,
    })?;
    let mut writer = DatasetWriter::create(&opts.out_dir, cfg)?;
    let mut map_entries = Vec::new();
    for (sha, (name, bytes)) in &maps_used {
        let file = format!("maps/{sha}.map");
        let p = opts.out_dir.join(&file);
        fs::write(&p, bytes).map_err(|source| DatasetError::Io { path: p, source })?;
        let loaded = ddai_map::load_map(bytes).map_err(|_| DatasetError::Map(sha.clone()))?;
        map_entries.push(MapEntry {
            sha256: sha.clone(),
            name: name.clone(),
            width: loaded.data.width,
            height: loaded.data.height,
            file,
        });
    }
    let mut totals = Counts {
        demos_total: counts.total,
        demos_ok: counts.ok,
        demos_skipped: counts.skipped,
        ..Counts::default()
    };
    for (di, out) in outputs.iter().enumerate() {
        if let Some(out) = out {
            writer.write_demo(di as u32, &out.frames, &out.samples)?;
            totals.frames += out.frames.len() as u64;
            totals.samples += out.samples.len() as u64;
            totals.confident_samples += out.samples.iter().filter(|s| s.confident()).count() as u64;
        }
    }
    totals.players = ranking.len() as u64;
    totals.chunks = writer.chunks.len() as u64;
    let manifest = Manifest {
        format: dataset::FORMAT_NAME.to_string(),
        format_version: FORMAT_VERSION,
        name: opts.name.clone(),
        source: opts.source.clone(),
        code_commit: opts.code_commit.clone(),
        config_hash: cfg.hash_hex(),
        config: cfg.clone(),
        demos: demos.clone(),
        maps: map_entries.clone(),
        chunks: writer.chunks.clone(),
        counts: totals,
    };
    write_json(&opts.out_dir.join(MANIFEST), &manifest)?;
    write_json(&opts.out_dir.join(PLAYERS), &ranking)?;

    let mut rep = report::build(&demos, &map_entries, &outputs, &ranking, counts, opts.top_players);
    rep.source = opts.source.clone();
    rep.code_commit = opts.code_commit.clone();
    rep.config_hash = cfg.hash_hex();
    write_json(&opts.out_dir.join(REPORT), &rep)?;
    log(&format!(
        "wrote {} chunks, {} frames, {} samples to {}",
        manifest.counts.chunks,
        manifest.counts.frames,
        manifest.counts.samples,
        opts.out_dir.display()
    ));
    Ok(rep)
}
