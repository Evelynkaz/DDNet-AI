//! Running a whole config and writing its files: one JSONL per condition, `summary.json`,
//! `summary.md`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::EnvError;
use crate::arena::Arena;
use crate::config::{BrainFactory, RunConfig};
use crate::game::GameReport;
use crate::report::{MapRecord, RunMeta, RunSummary, StallBaseline, markdown, summarize};
use crate::run::run_condition;

/// One JSONL line: the game record plus which condition and game index it belongs to.
#[derive(Serialize)]
pub struct Line<'a> {
    pub condition: &'a str,
    pub game: u32,
    #[serde(flatten)]
    pub report: &'a GameReport,
}

/// The git commit of the repository at `dir` and whether tracked files have uncommitted changes.
/// `("unknown", false)` when `git` is unavailable or `dir` is not a repository.
pub fn git_info(dir: &Path) -> (String, bool) {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    match run(&["rev-parse", "HEAD"]) {
        Some(commit) if !commit.is_empty() => {
            let dirty = run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
            (commit, dirty)
        }
        _ => ("unknown".to_string(), false),
    }
}

/// A file-name-safe version of a condition name.
pub fn slug(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-")
}

/// Inputs of [`run_config`] that are not part of the config file.
pub struct RunOptions<'a> {
    pub threads: usize,
    /// Write files here (created if needed); `None` = compute only.
    pub out_dir: Option<&'a Path>,
    /// Only conditions whose name contains this text.
    pub filter: Option<&'a str>,
    pub git: (String, bool),
    pub stall_baseline: Option<StallBaseline>,
}

/// Runs every (selected) condition of `cfg` and writes the outputs. `progress` gets one line per
/// finished condition.
pub fn run_config(
    cfg: &RunConfig,
    arenas: &BTreeMap<String, Arena>,
    factory: &BrainFactory,
    opts: &RunOptions<'_>,
    progress: &mut dyn FnMut(&str),
) -> Result<RunSummary, EnvError> {
    if let Some(dir) = opts.out_dir {
        std::fs::create_dir_all(dir).map_err(|e| EnvError::new(format!("creating {}: {e}", dir.display())))?;
    }
    let t0 = Instant::now();
    let mut conditions = Vec::new();
    let mut maps: BTreeMap<String, MapRecord> = BTreeMap::new();
    for cond in &cfg.condition {
        if opts.filter.is_some_and(|f| !cond.name.contains(f)) {
            continue;
        }
        let arena = arenas
            .get(&cond.arena)
            .ok_or_else(|| EnvError::new(format!("unknown arena {:?}", cond.arena)))?;
        let games = cfg.games_for(cond);
        let run = run_condition(cfg, cond, arena, games, factory, opts.threads)?;
        let summary = summarize(&run, arena.tag.label(), arena.map_sha256.clone());
        if let Some(sha) = &arena.map_sha256 {
            maps.entry(arena.name.clone()).or_insert_with(|| MapRecord {
                arena: arena.name.clone(),
                source: arena.map_source.clone(),
                sha256: sha.clone(),
            });
        }
        if let Some(dir) = opts.out_dir {
            let path = dir.join(format!("{}.jsonl", slug(&cond.name)));
            let file = std::fs::File::create(&path).map_err(|e| EnvError::new(format!("{}: {e}", path.display())))?;
            let mut w = std::io::BufWriter::new(file);
            for (g, report) in run.games.iter().enumerate() {
                let line = Line {
                    condition: &cond.name,
                    game: g as u32,
                    report,
                };
                serde_json::to_writer(&mut w, &line).map_err(|e| EnvError::new(e.to_string()))?;
                w.write_all(b"\n").map_err(|e| EnvError::new(e.to_string()))?;
            }
            w.flush().map_err(|e| EnvError::new(e.to_string()))?;
        }
        let t = summary.tally;
        progress(&format!(
            "{}: {} games {}:{}:{}:{} in {:.1}s ({:.1} games/s)",
            cond.name, summary.games, t.w, t.l, t.d, t.t, summary.wall_s, summary.games_per_s
        ));
        conditions.push(summary);
    }
    let summary = RunSummary {
        meta: RunMeta {
            run_name: cfg.name.clone(),
            config_hash: cfg.hash(),
            git_commit: opts.git.0.clone(),
            git_dirty: opts.git.1,
            base_seed: cfg.base_seed,
            threads: opts.threads,
            maps: maps.into_values().collect(),
            stall_baseline: opts.stall_baseline.clone(),
            total_wall_s: t0.elapsed().as_secs_f64(),
        },
        conditions,
    };
    if let Some(dir) = opts.out_dir {
        let json = serde_json::to_string_pretty(&summary).map_err(|e| EnvError::new(e.to_string()))?;
        std::fs::write(dir.join("summary.json"), json).map_err(|e| EnvError::new(e.to_string()))?;
        std::fs::write(dir.join("summary.md"), markdown(&summary)).map_err(|e| EnvError::new(e.to_string()))?;
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_is_file_name_safe() {
        assert_eq!(
            slug("CLB left: planner vs scripted (1v3)"),
            "clb-left-planner-vs-scripted-1v3"
        );
        assert_eq!(slug("--"), "");
    }
}
