//! The «Обучение» tab's data side (task 5.8, D-077, `docs/formats.md` §29): a **read-only** view of the training runs
//! under `<data-dir>/runs` (`--runs-dir`).
//!
//! * [`paths`] — what a request may name (identifiers only) and the check that every file lies under the runs root, symlinks
//!   resolved;
//! * [`read`] — bounded reads: whole small files or a refusal, the tail of the big append-only `metrics.jsonl`;
//! * [`parse`] — `status.json`, `metrics.jsonl`, `config.toml`, arena `summary.json` into the structures the page draws;
//! * this module — [`TrainingStore`]: the listing of experiments and runs, and the detail of one run (series, DAgger rounds,
//!   arena tables with Wilson intervals, checkpoints by name / size / sha256 prefix — never their bytes).
//!
//! The web never writes, moves or deletes anything here: no function in this module opens a file for writing, and the
//! production unit (`ProtectHome=read-only`, `User=ubuntu`) can read the tree as is (no unit change needed).
//!
//! Layout the scanner understands (`docs/formats.md` §22.4): `<runs>/<experiment>/<run>/{status.json, metrics.jsonl,
//! config*.toml, checkpoints/*.bundle, rounds/*.bundle}` and, for arena summaries, `<runs>/<experiment>/eval/<run>/<arena
//! dir>/summary.json` (the run's own name, or the name without the experiment prefix: `E-005/e005-fly` ↔ `eval/fly`). A
//! directory is a run when it has a `status.json` or a `metrics.jsonl`. Anything else in the tree is ignored.

pub mod parse;
pub mod paths;
pub mod read;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::Semaphore;

use self::parse::{ArenaSummary, ConfigSummary, Metrics, StatusInfo};
use self::paths::{PathError, RunsRoot, is_valid_ident};
use self::read::{ReadError, hash_file, read_capped, read_tail};

/// A job that has not touched `status.json` or `metrics.jsonl` for this long is shown as «не обновляется» rather than
/// «идёт». The data-collection step of a DAgger round writes nothing for 15–30 minutes (a loaded machine: an hour), so the
/// window is generous.
pub const ACTIVE_WINDOW_SECS: u64 = 2 * 3600;

/// Caps on what one request reads and sends.
#[derive(Debug, Clone)]
pub struct Limits {
    /// `status.json` is a few hundred bytes.
    pub status_bytes: u64,
    pub config_bytes: u64,
    /// How much of the end of `metrics.jsonl` is read (a real run: ~170 KB).
    pub metrics_tail_bytes: u64,
    pub max_line_bytes: usize,
    pub summary_bytes: u64,
    /// A checkpoint bigger than this is listed without a hash.
    pub hash_bytes: u64,
    pub max_experiments: usize,
    pub max_runs_per_experiment: usize,
    /// Directory entries looked at per directory (a hostile or huge directory cannot stall a request).
    pub max_dir_entries: usize,
    /// `train` points sent to the page.
    pub max_series_points: usize,
    /// Records kept per kind other than `train` (eval, arena, …): the newest. (`train` is bounded by the tail read and thinned
    /// to `max_series_points`.)
    pub max_records: usize,
    pub max_checkpoints: usize,
    pub max_eval_summaries: usize,
    pub max_conditions: usize,
    /// How many scans may touch the disk at once; more wait for a slot (they are not refused).
    pub max_concurrent_scans: usize,
    /// How long a request waits for a scan slot before it is answered `503 busy`.
    pub scan_wait: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            status_bytes: 16 * 1024,
            config_bytes: 64 * 1024,
            metrics_tail_bytes: 4 * 1024 * 1024,
            max_line_bytes: 256 * 1024,
            summary_bytes: 2 * 1024 * 1024,
            hash_bytes: 64 * 1024 * 1024,
            max_experiments: 200,
            max_runs_per_experiment: 400,
            max_dir_entries: 5000,
            max_series_points: 600,
            max_records: 2000,
            max_checkpoints: 100,
            max_eval_summaries: 8,
            max_conditions: 64,
            max_concurrent_scans: 4,
            scan_wait: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// The phase is not `done` and a file changed within [`ACTIVE_WINDOW_SECS`].
    Running,
    Done,
    /// The phase is not `done` but nothing was written for [`ACTIVE_WINDOW_SECS`] (stopped, killed or crashed).
    Stalled,
    /// No readable `status.json`.
    Unknown,
}

/// What the run list shows per run.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    pub id: String,
    pub kind: Option<String>,
    pub state: RunState,
    pub phase: Option<String>,
    pub step: Option<u64>,
    pub phase_step: Option<u64>,
    pub phase_steps: Option<u64>,
    pub planned_steps: Option<u64>,
    pub loss: Option<f64>,
    /// Seconds since the newest of `status.json` / `metrics.jsonl` changed.
    pub age_s: Option<u64>,
    pub has_metrics: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Experiment {
    pub id: String,
    pub runs: Vec<RunSummary>,
    pub runs_truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Listing {
    /// The runs directory exists and is readable.
    pub root_present: bool,
    pub experiments: Vec<Experiment>,
    pub experiments_truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Checkpoint {
    /// `checkpoints` or `rounds`.
    pub group: &'static str,
    pub name: String,
    pub bytes: u64,
    pub age_s: Option<u64>,
    /// First 16 hex characters of the file's sha256 (`None` for a file over the hash cap or one that could not be read).
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsOut {
    #[serde(flatten)]
    pub data: Metrics,
    /// Only the last bytes of a big file were read: earlier records are missing.
    pub tail_truncated: bool,
    pub file_len: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunDetail {
    pub exp: String,
    pub run: String,
    pub summary: RunSummary,
    pub status: Option<StatusInfo>,
    pub config: Option<ConfigSummary>,
    pub config_files: Vec<String>,
    pub metrics: Option<MetricsOut>,
    pub checkpoints: Vec<Checkpoint>,
    pub eval_summaries: Vec<ArenaSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DetailError {
    #[error("invalid name")]
    BadName,
    #[error("not found")]
    NotFound,
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Why [`TrainingStore::scan`] gave no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanError {
    /// No scan slot became free within [`Limits::scan_wait`].
    Busy,
    /// The scan task died (a panic) or the store is shutting down.
    Failed,
}

struct HashEntry {
    len: u64,
    mtime_nanos: u128,
    prefix: Option<String>,
}

pub struct TrainingStore {
    root: PathBuf,
    limits: Limits,
    /// `(len, mtime)` → sha256 prefix per checkpoint path: a checkpoint is hashed once, not on every poll.
    hashes: Mutex<HashMap<PathBuf, HashEntry>>,
    /// Bounds concurrent disk scans ([`Limits::max_concurrent_scans`]); see [`TrainingStore::scan`].
    gate: Arc<Semaphore>,
}

const HASH_CACHE_MAX: usize = 4096;
const HASH_PREFIX_LEN: usize = 16;

impl TrainingStore {
    pub fn new(root: PathBuf, limits: Limits) -> Self {
        Self {
            root,
            hashes: Mutex::new(HashMap::new()),
            gate: Arc::new(Semaphore::new(limits.max_concurrent_scans.max(1))),
            limits,
        }
    }

    /// Runs `work` (blocking disk reads) on the blocking pool once a scan slot is free. Requests beyond
    /// [`Limits::max_concurrent_scans`] **queue** (the page fires a list, a run and several comparison runs at once) and are
    /// refused with [`ScanError::Busy`] only after [`Limits::scan_wait`].
    pub async fn scan<T, F>(self: &Arc<Self>, work: F) -> Result<T, ScanError>
    where
        T: Send + 'static,
        F: FnOnce(&TrainingStore) -> T + Send + 'static,
    {
        let permit = match tokio::time::timeout(self.limits.scan_wait, Arc::clone(&self.gate).acquire_owned()).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => return Err(ScanError::Failed),
            Err(_) => return Err(ScanError::Busy),
        };
        let store = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            let out = work(&store);
            drop(permit);
            out
        })
        .await
        .map_err(|_| ScanError::Failed)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    // -----------------------------------------------------------------------------------------
    // Listing
    // -----------------------------------------------------------------------------------------

    /// Experiments and their runs. Blocking (run it on `spawn_blocking`).
    pub fn list(&self, now: u64) -> Listing {
        let Ok(root) = RunsRoot::open(&self.root) else {
            return Listing {
                root_present: false,
                experiments: Vec::new(),
                experiments_truncated: false,
            };
        };
        let mut names = subdir_names(root.path(), self.limits.max_dir_entries);
        names.sort_unstable_by(|a, b| b.cmp(a));
        let experiments_truncated = names.len() > self.limits.max_experiments;
        names.truncate(self.limits.max_experiments);

        let mut experiments = Vec::new();
        for exp in names {
            let Ok(exp_dir) = root.dir(&[&exp]) else { continue };
            let mut run_names = subdir_names(&exp_dir, self.limits.max_dir_entries);
            run_names.sort_unstable();
            let mut runs = Vec::new();
            let mut runs_truncated = false;
            for run in run_names {
                let Ok(run_dir) = root.dir(&[&exp, &run]) else { continue };
                let Some(summary) = self.summarize_run(&root, &run_dir, &run, now) else {
                    continue;
                };
                if runs.len() >= self.limits.max_runs_per_experiment {
                    runs_truncated = true;
                    break;
                }
                runs.push(summary);
            }
            if !runs.is_empty() {
                experiments.push(Experiment {
                    id: exp,
                    runs,
                    runs_truncated,
                });
            }
        }
        Listing {
            root_present: true,
            experiments,
            experiments_truncated,
        }
    }

    /// The list row of a run directory; `None` when it is not a run (no `status.json` and no `metrics.jsonl`).
    fn summarize_run(&self, root: &RunsRoot, dir: &Path, name: &str, now: u64) -> Option<RunSummary> {
        let status_path = root.file(dir, "status.json").ok();
        let metrics_path = root.file(dir, "metrics.jsonl").ok();
        if status_path.is_none() && metrics_path.is_none() {
            return None;
        }
        let status = status_path
            .as_ref()
            .and_then(|p| read_capped(p, self.limits.status_bytes).ok())
            .and_then(|b| parse::parse_status(&b));
        let config = self.read_config(root, dir);
        let mtimes = [status_path.as_ref(), metrics_path.as_ref()]
            .into_iter()
            .flatten()
            .filter_map(|p| mtime_secs(p));
        let last_update = mtimes.max();
        Some(build_summary(
            name,
            status.as_ref(),
            config.as_ref(),
            last_update,
            metrics_path.is_some(),
            now,
        ))
    }

    fn read_config(&self, root: &RunsRoot, dir: &Path) -> Option<ConfigSummary> {
        let path = root.file(dir, "config.toml").ok()?;
        let bytes = read_capped(&path, self.limits.config_bytes).ok()?;
        parse::parse_config(std::str::from_utf8(&bytes).ok()?)
    }

    // -----------------------------------------------------------------------------------------
    // One run
    // -----------------------------------------------------------------------------------------

    /// Everything about one run. Blocking.
    pub fn run_detail(&self, exp: &str, run: &str, now: u64) -> Result<RunDetail, DetailError> {
        if !is_valid_ident(exp) || !is_valid_ident(run) {
            return Err(DetailError::BadName);
        }
        let root = RunsRoot::open(&self.root).map_err(|_| DetailError::NotFound)?;
        let dir = root.dir(&[exp, run]).map_err(|e| match e {
            PathError::Invalid => DetailError::BadName,
            _ => DetailError::NotFound,
        })?;

        let status_path = root.file(&dir, "status.json").ok();
        let metrics_path = root.file(&dir, "metrics.jsonl").ok();
        if status_path.is_none() && metrics_path.is_none() {
            return Err(DetailError::NotFound);
        }
        let status = status_path
            .as_ref()
            .and_then(|p| read_capped(p, self.limits.status_bytes).ok())
            .and_then(|b| parse::parse_status(&b));
        let config = self.read_config(&root, &dir);
        let last_update = [status_path.as_ref(), metrics_path.as_ref()]
            .into_iter()
            .flatten()
            .filter_map(|p| mtime_secs(p))
            .max();
        let summary = build_summary(
            run,
            status.as_ref(),
            config.as_ref(),
            last_update,
            metrics_path.is_some(),
            now,
        );

        let metrics = metrics_path
            .as_ref()
            .and_then(|p| match read_tail(p, self.limits.metrics_tail_bytes) {
                Ok(tail) => Some(MetricsOut {
                    data: parse::parse_metrics(&tail.bytes, &self.limits),
                    tail_truncated: tail.truncated,
                    file_len: tail.file_len,
                }),
                Err(_) => None,
            });

        Ok(RunDetail {
            exp: exp.to_string(),
            run: run.to_string(),
            summary,
            status,
            config,
            config_files: self.config_files(&dir),
            metrics,
            checkpoints: self.checkpoints(&root, exp, run, now),
            eval_summaries: self.eval_summaries(&root, exp, run),
        })
    }

    fn config_files(&self, dir: &Path) -> Vec<String> {
        let mut files: Vec<String> = file_names(dir, self.limits.max_dir_entries)
            .into_iter()
            .filter(|n| n.starts_with("config") && n.ends_with(".toml"))
            .collect();
        files.sort_unstable();
        files.truncate(20);
        files
    }

    fn checkpoints(&self, root: &RunsRoot, exp: &str, run: &str, now: u64) -> Vec<Checkpoint> {
        let mut out = Vec::new();
        for group in ["checkpoints", "rounds"] {
            let Ok(dir) = root.dir(&[exp, run, group]) else {
                continue;
            };
            let mut names: Vec<String> = file_names(&dir, self.limits.max_dir_entries)
                .into_iter()
                .filter(|n| n.ends_with(".bundle"))
                .collect();
            names.sort_unstable();
            for name in names {
                if out.len() >= self.limits.max_checkpoints {
                    return out;
                }
                let Ok(path) = root.file(&dir, &name) else { continue };
                let Ok(meta) = std::fs::metadata(&path) else { continue };
                let mtime = meta.modified().ok();
                out.push(Checkpoint {
                    group: if group == "rounds" { "rounds" } else { "checkpoints" },
                    name,
                    bytes: meta.len(),
                    age_s: mtime
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| now.saturating_sub(d.as_secs())),
                    sha256: self.sha_prefix(&path, meta.len(), mtime),
                });
            }
        }
        out
    }

    fn sha_prefix(&self, path: &Path, len: u64, mtime: Option<SystemTime>) -> Option<String> {
        let stamp = mtime
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        if let Ok(cache) = self.hashes.lock()
            && let Some(entry) = cache.get(path)
            && entry.len == len
            && entry.mtime_nanos == stamp
        {
            return entry.prefix.clone();
        }
        let hash = match hash_file(path, self.limits.hash_bytes) {
            Ok(full) => Some(full[..HASH_PREFIX_LEN].to_string()),
            Err(_) => None,
        };
        if let Ok(mut cache) = self.hashes.lock() {
            if cache.len() >= HASH_CACHE_MAX {
                cache.clear();
            }
            cache.insert(
                path.to_path_buf(),
                HashEntry {
                    len,
                    mtime_nanos: stamp,
                    prefix: hash.clone(),
                },
            );
        }
        hash
    }

    fn eval_summaries(&self, root: &RunsRoot, exp: &str, run: &str) -> Vec<ArenaSummary> {
        let Some(eval_dir_name) = eval_dir_candidates(exp, run)
            .into_iter()
            .find(|c| root.dir(&[exp, "eval", c]).is_ok())
        else {
            return Vec::new();
        };
        let Ok(eval_dir) = root.dir(&[exp, "eval", &eval_dir_name]) else {
            return Vec::new();
        };
        let mut sources = subdir_names(&eval_dir, self.limits.max_dir_entries);
        sources.sort_unstable();
        let mut out = Vec::new();
        for source in sources {
            if out.len() >= self.limits.max_eval_summaries {
                break;
            }
            let Ok(dir) = root.dir(&[exp, "eval", &eval_dir_name, &source]) else {
                continue;
            };
            let Ok(path) = root.file(&dir, "summary.json") else {
                continue;
            };
            match read_capped(&path, self.limits.summary_bytes) {
                Ok(bytes) => {
                    if let Some(s) = parse::parse_arena_summary(&bytes, &source, self.limits.max_conditions) {
                        out.push(s);
                    }
                }
                Err(ReadError::TooLarge | ReadError::NotFile | ReadError::Io) => {}
            }
        }
        out
    }
}

/// The directory names under `<exp>/eval/` that belong to a run: its own name, or the name without the experiment prefix
/// (`E-005` + `e005-fly` → `fly`).
pub fn eval_dir_candidates(exp: &str, run: &str) -> Vec<String> {
    let mut out = vec![run.to_string()];
    let prefix = format!("{}-", exp.to_ascii_lowercase().replace('-', ""));
    if let Some(rest) = run.strip_prefix(&prefix)
        && !rest.is_empty()
    {
        out.push(rest.to_string());
    }
    out
}

fn build_summary(
    name: &str,
    status: Option<&StatusInfo>,
    config: Option<&ConfigSummary>,
    last_update: Option<u64>,
    has_metrics: bool,
    now: u64,
) -> RunSummary {
    let phase = status.and_then(|s| s.phase.clone());
    let age_s = last_update.map(|t| now.saturating_sub(t));
    RunSummary {
        id: name.to_string(),
        kind: config.and_then(|c| c.kind.clone()),
        state: classify(phase.as_deref(), age_s),
        phase,
        step: status.and_then(|s| s.step),
        phase_step: status.and_then(|s| s.phase_step),
        phase_steps: status.and_then(|s| s.phase_steps),
        planned_steps: config.and_then(|c| c.planned_steps),
        loss: status.and_then(|s| s.loss),
        age_s,
        has_metrics,
    }
}

/// `done` is final; anything else is `running` while the files are fresh and `stalled` after [`ACTIVE_WINDOW_SECS`].
pub fn classify(phase: Option<&str>, age_s: Option<u64>) -> RunState {
    match (phase, age_s) {
        (Some("done"), _) => RunState::Done,
        (Some(_), Some(age)) if age <= ACTIVE_WINDOW_SECS => RunState::Running,
        (Some(_), Some(_)) => RunState::Stalled,
        _ => RunState::Unknown,
    }
}

fn mtime_secs(path: &Path) -> Option<u64> {
    let t = std::fs::metadata(path).ok()?.modified().ok()?;
    t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

/// Names of the entries of `dir` that are valid identifiers and directories (or symlinks, resolved later by
/// [`RunsRoot::dir`]). At most `limit` entries are looked at.
fn subdir_names(dir: &Path, limit: usize) -> Vec<String> {
    list_names(dir, limit, |t| t.is_dir() || t.is_symlink())
}

/// Like [`subdir_names`] for regular files (and symlinks, resolved later by [`RunsRoot::file`]).
fn file_names(dir: &Path, limit: usize) -> Vec<String> {
    list_names(dir, limit, |t| t.is_file() || t.is_symlink())
}

fn list_names(dir: &Path, limit: usize, keep: impl Fn(&std::fs::FileType) -> bool) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.take(limit)
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| keep(&t)))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| is_valid_ident(n))
        .collect()
}

#[cfg(test)]
mod tests;
