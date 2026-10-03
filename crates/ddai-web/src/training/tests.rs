//! Tests of the store on scratch run directories with synthetic data (no real run is touched).

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;

use super::*;

const NOW: u64 = 1_800_000_000;

fn store(root: &Path) -> TrainingStore {
    TrainingStore::new(root.to_path_buf(), Limits::default())
}

fn write(path: &Path, content: impl AsRef<[u8]>) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn jsonl(rows: &[serde_json::Value]) -> String {
    rows.iter().map(|r| r.to_string() + "\n").collect()
}

fn metrics_rows() -> Vec<serde_json::Value> {
    vec![
        json!({"kind":"train","phase":"bc","step":50,"loss":{"total":3.0,"dir":1.0,"hook":0.6}}),
        json!({"kind":"train","phase":"bc","step":100,"loss":{"total":2.0,"dir":0.9,"hook":0.5}}),
        json!({"kind":"eval","phase":"bc","set":"dagger-val","step":100,"report":{"dir":{"accuracy":0.7},"hook":{"auroc":0.8}}}),
        json!({"kind":"arena","phase":"bc","step":100,"eval":{"arena":"clb-left","games":300,"w":100,"l":150,"d":0,"t":50,
            "credited_w":30,"credited_win_rate":[0.1,0.07,0.14],"win_rate":[0.4,0.35,0.45],"win_rate_all":[0.33,0.28,0.38]}}),
        json!({"kind":"train","phase":"dagger-1","step":150,"loss":{"total":1.5}}),
    ]
}

const CONFIG: &str = "bc_steps = 100\n[model]\nkind = \"fly\"\n[dagger]\nbetas = [0.5, 0.4]\nsteps_per_round = 50\n";

fn make_run(root: &Path, exp: &str, run: &str, status: &str) -> PathBuf {
    let dir = root.join(exp).join(run);
    write(&dir.join("status.json"), status);
    write(&dir.join("metrics.jsonl"), jsonl(&metrics_rows()));
    write(&dir.join("config.toml"), CONFIG);
    dir
}

#[test]
fn listing_finds_runs_with_their_states_and_ignores_everything_else() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    make_run(root, "E-1", "run-done", r#"{"phase":"done","step":150}"#);
    make_run(
        root,
        "E-1",
        "run-live",
        r#"{"phase":"dagger-1","step":150,"phase_step":10,"phase_steps":50,"loss":1.5}"#,
    );
    make_run(root, "E-2", "other", r#"{"phase":"bc","step":50}"#);
    // Not runs: a dir without status/metrics, a loose file, a hidden dir, a name that is not an identifier, an empty exp.
    fs::create_dir_all(root.join("E-1").join("eval").join("x")).unwrap();
    write(&root.join("E-1").join("notes.txt"), "x");
    fs::create_dir_all(root.join("E-1").join(".hidden")).unwrap();
    fs::create_dir_all(root.join("E-1").join("bad name")).unwrap();
    fs::create_dir_all(root.join("E-3-empty").join("nothing")).unwrap();
    write(&root.join("stray.json"), "{}");
    // A metrics-only run (no status.json) is a run too.
    write(&root.join("E-2").join("metrics-only").join("metrics.jsonl"), "");

    let store = store(root);
    // `now` is just after the files were written: live runs are running.
    let now = unix_now();
    let listing = store.list(now);
    assert!(listing.root_present);
    let ids: Vec<_> = listing.experiments.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, ["E-2", "E-1"], "newest id first, empty experiments omitted");
    let e1 = &listing.experiments[1];
    assert_eq!(
        e1.runs.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["run-done", "run-live"]
    );
    assert_eq!(e1.runs[0].state, RunState::Done);
    assert_eq!(e1.runs[1].state, RunState::Running);
    assert_eq!(e1.runs[1].phase.as_deref(), Some("dagger-1"));
    assert_eq!(e1.runs[1].phase_step, Some(10));
    assert_eq!(e1.runs[1].planned_steps, Some(200));
    assert_eq!(e1.runs[1].kind.as_deref(), Some("fly"));
    let e2 = &listing.experiments[0];
    assert_eq!(e2.runs.len(), 2);
    let metrics_only = e2.runs.iter().find(|r| r.id == "metrics-only").unwrap();
    assert_eq!(metrics_only.state, RunState::Unknown);
    assert!(metrics_only.has_metrics);

    // Much later the unfinished run is not shown as running any more; a finished one stays done.
    let later = store.list(now + ACTIVE_WINDOW_SECS + 60);
    let e1 = later.experiments.iter().find(|e| e.id == "E-1").unwrap();
    assert_eq!(e1.runs[0].state, RunState::Done);
    assert_eq!(e1.runs[1].state, RunState::Stalled);
}

#[test]
fn a_missing_runs_root_is_an_empty_listing_not_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let listing = store(&tmp.path().join("absent")).list(NOW);
    assert!(!listing.root_present);
    assert!(listing.experiments.is_empty());
}

#[test]
fn classify_covers_every_combination() {
    assert_eq!(classify(Some("done"), Some(1_000_000)), RunState::Done);
    assert_eq!(classify(Some("done"), None), RunState::Done);
    assert_eq!(classify(Some("bc"), Some(0)), RunState::Running);
    assert_eq!(classify(Some("bc"), Some(ACTIVE_WINDOW_SECS)), RunState::Running);
    assert_eq!(classify(Some("bc"), Some(ACTIVE_WINDOW_SECS + 1)), RunState::Stalled);
    assert_eq!(classify(Some("bc"), None), RunState::Unknown);
    assert_eq!(classify(None, Some(5)), RunState::Unknown);
}

#[test]
fn run_detail_returns_series_checkpoints_and_eval_summaries() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let dir = make_run(root, "E-005", "e005-fly", r#"{"phase":"done","step":150}"#);
    write(&dir.join("config-before-1.toml"), CONFIG);
    write(&dir.join("checkpoints").join("final.bundle"), b"abc");
    write(
        &dir.join("checkpoints").join("step-00000100.bundle"),
        b"weights-weights-weights",
    );
    write(&dir.join("checkpoints").join("notes.txt"), b"not a checkpoint");
    write(&dir.join("rounds").join("round-0.bundle"), b"round");
    write(&dir.join("state.bin"), b"state");
    let summary = json!({"meta":{"git_commit":"0123456789abcdef","base_seed":1},"conditions":[
        {"name":"clb-left vs scripted","arena":"clb-left","games":100,"tally":{"w":50,"l":40,"d":0,"t":10},
         "win_rate":{"p":0.55,"lo":0.45,"hi":0.64},"win_rate_all":{"p":0.5,"lo":0.4,"hi":0.6},"credited_w":5}]});
    write(
        &root
            .join("E-005")
            .join("eval")
            .join("fly")
            .join("arena")
            .join("summary.json"),
        summary.to_string(),
    );
    write(
        &root
            .join("E-005")
            .join("eval")
            .join("fly")
            .join("scenarios")
            .join("scenarios.json"),
        "{}",
    );

    let detail = store(root).run_detail("E-005", "e005-fly", NOW).unwrap();
    assert_eq!(detail.summary.state, RunState::Done);
    assert_eq!(detail.config_files, ["config-before-1.toml", "config.toml"]);
    let m = detail.metrics.as_ref().unwrap();
    assert!(!m.tail_truncated);
    assert_eq!(m.data.train.len(), 3);
    assert_eq!(
        m.data.phases.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        ["bc", "dagger-1"]
    );
    assert_eq!(m.data.arena[0].credited.unwrap().p, 0.1);

    let names: Vec<_> = detail
        .checkpoints
        .iter()
        .map(|c| format!("{}/{}", c.group, c.name))
        .collect();
    assert_eq!(
        names,
        [
            "checkpoints/final.bundle",
            "checkpoints/step-00000100.bundle",
            "rounds/round-0.bundle"
        ]
    );
    let final_ck = &detail.checkpoints[0];
    assert_eq!(final_ck.bytes, 3);
    assert_eq!(final_ck.sha256.as_deref(), Some("ba7816bf8f01cfea"));

    assert_eq!(detail.eval_summaries.len(), 1, "scenarios.json is not an arena summary");
    assert_eq!(detail.eval_summaries[0].source, "arena");
    assert_eq!(detail.eval_summaries[0].git_commit.as_deref(), Some("0123456789ab"));
    assert_eq!(
        detail.eval_summaries[0].conditions[0].credited.unwrap(),
        parse::wilson(5, 100).unwrap()
    );

    // The JSON of the whole detail never contains a checkpoint's bytes, nor a path of the machine.
    let json = serde_json::to_string(&detail).unwrap();
    assert!(!json.contains("weights-weights"));
    assert!(!json.contains(root.to_str().unwrap()));
}

#[test]
fn eval_summaries_match_the_run_name_or_the_name_without_the_experiment_prefix() {
    assert_eq!(
        eval_dir_candidates("E-005", "e005-fly-noclb"),
        ["e005-fly-noclb", "fly-noclb"]
    );
    assert_eq!(
        eval_dir_candidates("E-008", "e008-p1-fly-drop-s1"),
        ["e008-p1-fly-drop-s1", "p1-fly-drop-s1"]
    );
    assert_eq!(eval_dir_candidates("E-1", "plain"), ["plain"]);
    assert_eq!(eval_dir_candidates("E-1", "e1-"), ["e1-"]);
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    make_run(root, "E-8", "e008-x", r#"{"phase":"done","step":1}"#);
    let summary = json!({"conditions":[{"name":"c","games":10,"credited_w":1,"tally":{"w":1,"l":9,"d":0,"t":0}}]});
    write(
        &root
            .join("E-8")
            .join("eval")
            .join("e008-x")
            .join("arena")
            .join("summary.json"),
        summary.to_string(),
    );
    let detail = store(root).run_detail("E-8", "e008-x", NOW).unwrap();
    assert_eq!(detail.eval_summaries.len(), 1);
}

#[test]
fn traversal_and_unknown_names_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("runs");
    make_run(&root, "E-1", "run-a", r#"{"phase":"done","step":1}"#);
    write(
        &tmp.path().join("outside").join("run-x").join("status.json"),
        r#"{"phase":"done","step":1}"#,
    );
    let store = store(&root);
    for (exp, run) in [
        ("..", "outside"),
        ("E-1", ".."),
        ("E-1", "../E-1/run-a"),
        ("E-1/run-a", "x"),
        ("", "run-a"),
        ("E-1", ""),
        ("E-1", "run a"),
        ("/etc", "passwd"),
        ("E-1", "run-a/"),
    ] {
        assert_eq!(
            store.run_detail(exp, run, NOW).unwrap_err(),
            DetailError::BadName,
            "{exp:?} {run:?}"
        );
    }
    assert_eq!(store.run_detail("E-1", "nope", NOW).unwrap_err(), DetailError::NotFound);
    assert_eq!(
        store.run_detail("E-9", "run-a", NOW).unwrap_err(),
        DetailError::NotFound
    );
}

#[cfg(unix)]
#[test]
fn symlinks_out_of_the_root_show_nothing() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("runs");
    fs::create_dir_all(root.join("E-1")).unwrap();
    // A whole run directory elsewhere, linked in.
    let elsewhere = tmp.path().join("elsewhere").join("run-x");
    write(&elsewhere.join("status.json"), r#"{"phase":"done","step":1}"#);
    symlink(&elsewhere, root.join("E-1").join("linked-run")).unwrap();
    // A run whose metrics file is a link to a file outside.
    let run = make_run(&root, "E-1", "run-b", r#"{"phase":"done","step":1}"#);
    write(
        &tmp.path().join("elsewhere").join("secret.jsonl"),
        jsonl(&metrics_rows()),
    );
    fs::remove_file(run.join("metrics.jsonl")).unwrap();
    symlink(
        tmp.path().join("elsewhere").join("secret.jsonl"),
        run.join("metrics.jsonl"),
    )
    .unwrap();
    // A checkpoint link to a file outside.
    write(&tmp.path().join("elsewhere").join("big.bundle"), b"outside-bytes");
    fs::create_dir_all(run.join("checkpoints")).unwrap();
    symlink(
        tmp.path().join("elsewhere").join("big.bundle"),
        run.join("checkpoints").join("evil.bundle"),
    )
    .unwrap();
    // An experiment that is a link out.
    symlink(tmp.path().join("elsewhere"), root.join("E-link")).unwrap();

    let store = store(&root);
    assert_eq!(
        store.run_detail("E-1", "linked-run", NOW).unwrap_err(),
        DetailError::NotFound
    );
    assert_eq!(
        store.run_detail("E-link", "run-x", NOW).unwrap_err(),
        DetailError::NotFound
    );
    let detail = store.run_detail("E-1", "run-b", NOW).unwrap();
    assert!(detail.metrics.is_none(), "an escaping metrics.jsonl is not read");
    assert!(detail.checkpoints.is_empty(), "an escaping checkpoint is not listed");
    let listing = store.list(NOW);
    let ids: Vec<_> = listing.experiments.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, ["E-1"], "the escaping experiment is not listed");
    assert_eq!(
        listing.experiments[0]
            .runs
            .iter()
            .map(|r| r.id.as_str())
            .collect::<Vec<_>>(),
        ["run-b"]
    );
}

#[test]
fn oversized_files_are_capped_not_read() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let dir = root.join("E-1").join("big");
    // A status and a config past their caps are ignored (the run is still listed, with an unknown state).
    write(
        &dir.join("status.json"),
        format!("{{\"phase\":\"done\",\"pad\":\"{}\"}}", "x".repeat(2000)),
    );
    write(
        &dir.join("config.toml"),
        format!("# {}\nbc_steps = 1\n", "x".repeat(2000)),
    );
    // A metrics file far past the tail cap: only the last lines are read.
    let mut rows = String::new();
    for step in 1..=2000u64 {
        rows += &json!({"kind":"train","phase":"bc","step":step,"loss":{"total":1.0}}).to_string();
        rows.push('\n');
    }
    write(&dir.join("metrics.jsonl"), &rows);
    let limits = Limits {
        status_bytes: 1024,
        config_bytes: 1024,
        metrics_tail_bytes: 4096,
        ..Limits::default()
    };
    let store = TrainingStore::new(root.to_path_buf(), limits);
    let detail = store.run_detail("E-1", "big", NOW).unwrap();
    assert!(detail.status.is_none() && detail.config.is_none());
    assert_eq!(detail.summary.state, RunState::Unknown);
    let m = detail.metrics.unwrap();
    assert!(m.tail_truncated);
    assert_eq!(m.file_len, rows.len() as u64);
    assert!(
        !m.data.train.is_empty() && m.data.train.len() < 100,
        "{}",
        m.data.train.len()
    );
    assert_eq!(m.data.train.last().unwrap().step, 2000);
    assert_eq!(m.data.skipped_lines, 0, "the tail starts on a line boundary");
    let listing = store.list(NOW);
    assert_eq!(listing.experiments[0].runs[0].state, RunState::Unknown);
}

#[test]
fn a_huge_checkpoint_is_listed_without_a_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = make_run(tmp.path(), "E-1", "r", r#"{"phase":"done","step":1}"#);
    write(&dir.join("checkpoints").join("huge.bundle"), vec![7u8; 5000]);
    write(&dir.join("checkpoints").join("small.bundle"), b"abc");
    let limits = Limits {
        hash_bytes: 100,
        ..Limits::default()
    };
    let store = TrainingStore::new(tmp.path().to_path_buf(), limits);
    let detail = store.run_detail("E-1", "r", NOW).unwrap();
    let huge = detail.checkpoints.iter().find(|c| c.name == "huge.bundle").unwrap();
    assert_eq!((huge.bytes, huge.sha256.as_deref()), (5000, None));
    let small = detail.checkpoints.iter().find(|c| c.name == "small.bundle").unwrap();
    assert!(small.sha256.is_some());
}

#[test]
fn the_hash_cache_follows_changes_of_a_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = make_run(tmp.path(), "E-1", "r", r#"{"phase":"done","step":1}"#);
    let path = dir.join("checkpoints").join("last.bundle");
    write(&path, b"abc");
    let store = store(tmp.path());
    let first = store.run_detail("E-1", "r", NOW).unwrap().checkpoints[0].sha256.clone();
    assert_eq!(first.as_deref(), Some("ba7816bf8f01cfea"));
    let again = store.run_detail("E-1", "r", NOW).unwrap().checkpoints[0].sha256.clone();
    assert_eq!(again, first);
    // The job rewrites the checkpoint (different length): the cache entry no longer matches.
    write(&path, b"abcd");
    let changed = store.run_detail("E-1", "r", NOW).unwrap().checkpoints[0].sha256.clone();
    assert_ne!(changed, first);
    assert_eq!(changed.as_deref(), Some("88d4266fd4e6338d"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scans_queue_for_a_slot_instead_of_being_refused() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    let tmp = tempfile::tempdir().unwrap();
    let limits = Limits {
        max_concurrent_scans: 2,
        scan_wait: Duration::from_secs(5),
        ..Limits::default()
    };
    let store = Arc::new(TrainingStore::new(tmp.path().to_path_buf(), limits));
    let running = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..10usize {
        let (store, running, peak) = (Arc::clone(&store), Arc::clone(&running), Arc::clone(&peak));
        tasks.push(tokio::spawn(async move {
            store
                .scan(move |_| {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(40));
                    running.fetch_sub(1, Ordering::SeqCst);
                    i
                })
                .await
        }));
    }
    let mut got = Vec::new();
    for t in tasks {
        got.push(t.await.unwrap().expect("a queued scan is answered, not refused"));
    }
    got.sort_unstable();
    assert_eq!(got, (0..10).collect::<Vec<_>>());
    assert_eq!(peak.load(Ordering::SeqCst), 2, "the cap still holds");
    assert!(
        started.elapsed() >= Duration::from_millis(180),
        "10 scans x 40 ms on 2 slots take at least 200 ms"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_scan_that_waits_too_long_for_a_slot_is_busy() {
    use std::sync::Arc;
    use std::time::Duration;
    let tmp = tempfile::tempdir().unwrap();
    let limits = Limits {
        max_concurrent_scans: 1,
        scan_wait: Duration::from_millis(80),
        ..Limits::default()
    };
    let store = Arc::new(TrainingStore::new(tmp.path().to_path_buf(), limits));
    let holder = {
        let store = Arc::clone(&store);
        tokio::spawn(async move { store.scan(|_| std::thread::sleep(Duration::from_millis(500))).await })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(store.scan(|_| ()).await, Err(ScanError::Busy));
    assert_eq!(holder.await.unwrap(), Ok(()));
    assert_eq!(store.scan(|_| 7).await, Ok(7), "the slot is free again");
}

/// Every (path, len, mtime) under `root`, for the "nothing was written" check.
fn snapshot(root: &Path) -> Vec<(PathBuf, u64, Option<std::time::SystemTime>)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        out.push((dir.clone(), 0, fs::metadata(&dir).unwrap().modified().ok()));
        for entry in fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let meta = entry.metadata().unwrap();
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                out.push((entry.path(), meta.len(), meta.modified().ok()));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn reading_never_changes_the_runs_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = make_run(tmp.path(), "E-1", "r", r#"{"phase":"dagger-1","step":150}"#);
    write(&dir.join("checkpoints").join("last.bundle"), b"abc");
    write(
        &tmp.path()
            .join("E-1")
            .join("eval")
            .join("r")
            .join("arena")
            .join("summary.json"),
        r#"{"conditions":[]}"#,
    );
    let before = snapshot(tmp.path());
    let store = store(tmp.path());
    for _ in 0..3 {
        store.list(NOW);
        store.run_detail("E-1", "r", NOW).unwrap();
        let _ = store.run_detail("E-1", "missing", NOW);
        let _ = store.run_detail("..", "r", NOW);
    }
    assert_eq!(snapshot(tmp.path()), before);
}

#[test]
fn a_run_being_written_while_read_gives_a_partial_last_line_that_is_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("E-1").join("r");
    write(&dir.join("status.json"), r#"{"phase":"bc","step":100}"#);
    let mut rows = jsonl(&metrics_rows());
    rows.push_str("{\"kind\":\"train\",\"phase\":\"dagger-1\",\"step\":2"); // an append caught half way
    write(&dir.join("metrics.jsonl"), rows);
    let detail = store(tmp.path()).run_detail("E-1", "r", NOW).unwrap();
    let m = detail.metrics.unwrap();
    assert_eq!(m.data.skipped_lines, 1);
    assert_eq!(m.data.train.len(), 3);
}

#[test]
fn the_run_json_carries_no_nickname_field_at_all() {
    // The panel's types have no place for a player name; the arena summary's `players` are dropped by the parser.
    let tmp = tempfile::tempdir().unwrap();
    let dir = make_run(tmp.path(), "E-1", "r", r#"{"phase":"done","step":1}"#);
    let summary = json!({"conditions":[{"name":"c","games":10,"credited_w":1,"players":[{"name":"SomeNick","clan":"Clan"}],
        "tally":{"w":1,"l":9,"d":0,"t":0}}]});
    write(
        &dir.parent()
            .unwrap()
            .join("eval")
            .join("r")
            .join("arena")
            .join("summary.json"),
        summary.to_string(),
    );
    let json = serde_json::to_string(&store(tmp.path()).run_detail("E-1", "r", NOW).unwrap()).unwrap();
    assert!(!json.contains("SomeNick") && !json.contains("Clan"));
}
