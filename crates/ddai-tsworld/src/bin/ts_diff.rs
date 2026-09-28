//! `ts-diff` — task 1.9 acceptance criterion 5's diagnostics helper: replays one trace-ts v1 file
//! (or every `*.jsonl` file under a corpus directory) against `ddai_tsworld::SimWorld` and reports
//! the first field-level mismatch, if any, plus (in `--corpus` mode) aggregate tile/event
//! coverage across the whole corpus.
//!
//! Usage:
//!   ts-diff <trace.jsonl>                  # kind auto-detected from the metadata line
//!   ts-diff --corpus <dir>                 # every *.jsonl under dir (recursively)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ddai_tsworld::trace_ts::{ReplayReport, replay_episode, replay_opscript};

fn trace_kind(jsonl: &str) -> String {
    let first_line = jsonl.lines().next().unwrap_or("{}");
    let v: serde_json::Value = serde_json::from_str(first_line).unwrap_or_default();
    v.get("kind").and_then(|k| k.as_str()).unwrap_or("episode").to_string()
}

fn replay_file(path: &Path) -> (String, ReplayReport) {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let kind = trace_kind(&content);
    let report = if kind == "opscript" {
        replay_opscript(&content)
    } else {
        replay_episode(&content)
    };
    (kind, report)
}

/// Coarse per-trace event/tile coverage counters (acceptance criterion 4's report: "freeze
/// entries, hammer hits, hook grabs, player hooks, deaths, respawns, projectiles, tele,
/// speedups"). Counted from the trace's own recorded `events`/`state`, independent of whether the
/// replay matched — coverage is about what the *corpus* exercises, not about parity per se.
#[derive(Default, Debug)]
struct Coverage {
    ticks_or_ops: u64,
    freeze_events: u64,
    hammer_fire_events: u64,
    hammer_hit_events: u64,
    hook_attach_ground: u64,
    death_events: u64,
    respawns_observed: u64,
    max_projectiles_seen: u64,
    max_lasers_seen: u64,
}

fn scan_coverage(jsonl: &str, cov: &mut Coverage) {
    let mut prev_alive: HashMap<i32, bool> = HashMap::new();
    for line in jsonl.lines().skip(1) {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        cov.ticks_or_ops += 1;
        if let Some(events) = v.get("events").and_then(|e| e.as_array()) {
            for e in events {
                match e.get("kind").and_then(|k| k.as_str()) {
                    Some("freeze") => cov.freeze_events += 1,
                    Some("hammerFire") => cov.hammer_fire_events += 1,
                    Some("hammerHit") => cov.hammer_hit_events += 1,
                    Some("death") => cov.death_events += 1,
                    _ => {}
                }
            }
        }
        if let Some(state) = v.get("state") {
            if let Some(tees) = state.get("tees").and_then(|t| t.as_array()) {
                for pair in tees {
                    let Some(arr) = pair.as_array() else { continue };
                    let id = arr[0].as_i64().unwrap_or(-1) as i32;
                    let core = &arr[1]["core"];
                    if core.get("hookState").and_then(|h| h.as_i64()) == Some(5) {
                        cov.hook_attach_ground += 1;
                    }
                    let alive = arr[1].get("alive").and_then(|a| a.as_bool()).unwrap_or(true);
                    if let Some(&was_alive) = prev_alive.get(&id)
                        && !was_alive
                        && alive
                    {
                        cov.respawns_observed += 1;
                    }
                    prev_alive.insert(id, alive);
                }
            }
            if let Some(p) = state.get("projectiles").and_then(|p| p.as_array()) {
                cov.max_projectiles_seen = cov.max_projectiles_seen.max(p.len() as u64);
            }
            if let Some(l) = state.get("lasers").and_then(|l| l.as_array()) {
                cov.max_lasers_seen = cov.max_lasers_seen.max(l.len() as u64);
            }
        }
    }
}

fn walk_jsonl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_jsonl(&path, out);
        } else if path.extension().is_some_and(|e| e == "jsonl") {
            out.push(path);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: ts-diff <trace.jsonl> | --corpus <dir>");
        std::process::exit(2);
    }

    if args[0] == "--corpus" {
        let dir = PathBuf::from(args.get(1).expect("--corpus needs a directory"));
        let mut files = Vec::new();
        walk_jsonl(&dir, &mut files);
        files.sort();
        println!("corpus: {} files under {}", files.len(), dir.display());

        let mut total_steps = 0u64;
        let mut total_mismatches = 0u64;
        let mut cov = Coverage::default();
        let mut first_reported = false;

        for path in &files {
            let content = std::fs::read_to_string(path).unwrap();
            scan_coverage(&content, &mut cov);
            let (_kind, report) = replay_file(path);
            total_steps += report.steps_replayed as u64;
            if let Some(m) = &report.first_mismatch {
                total_mismatches += 1;
                if !first_reported {
                    println!("FIRST MISMATCH in {}: {}", path.display(), m);
                    first_reported = true;
                }
            }
        }

        println!("--- summary ---");
        println!("files: {}", files.len());
        println!("total steps (ticks/ops) replayed: {total_steps}");
        println!("files with >=1 mismatch: {total_mismatches}");
        println!("coverage across corpus:");
        println!("  ticks/ops scanned:   {}", cov.ticks_or_ops);
        println!("  freeze events:       {}", cov.freeze_events);
        println!("  hammer fire events:  {}", cov.hammer_fire_events);
        println!("  hammer hit events:   {}", cov.hammer_hit_events);
        println!("  hook-grabbed ticks:  {}", cov.hook_attach_ground);
        println!("  death events:        {}", cov.death_events);
        println!("  respawns observed:   {}", cov.respawns_observed);
        println!("  max projectiles live:{}", cov.max_projectiles_seen);
        println!("  max lasers live:     {}", cov.max_lasers_seen);

        if total_mismatches > 0 {
            std::process::exit(1);
        }
        return;
    }

    let path = PathBuf::from(&args[0]);
    let (kind, report) = replay_file(&path);
    println!("kind: {kind}");
    println!("steps_replayed: {}", report.steps_replayed);
    match report.first_mismatch {
        Some(m) => {
            println!("MISMATCH: {m}");
            std::process::exit(1);
        }
        None => println!("OK: 0 mismatches"),
    }
}
