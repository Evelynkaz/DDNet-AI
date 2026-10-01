//! A whole tiny experiment, end to end: teacher data collected into a dataset, behaviour cloning,
//! one DAgger round (the student plays, the teacher labels, the data is aggregated, the model is
//! retrained), the run directory's files, and resuming a finished run (which must not redo
//! anything).

use std::path::PathBuf;

use ddai_env::models::ModelBrains;
use ddai_train::experiment::{JobConfig, load_env, run_collect_jobs};
use ddai_train::runner::{DaggerConfig, ExperimentConfig, ModelSpec, run_experiment};
use ddai_train::store::TeacherStore;
use ddai_train::teacher_data::TeacherDataConfig;
use ddai_train::trainer::TrainConfig;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn job(base_seed: u64, games: u32) -> JobConfig {
    JobConfig {
        arena: "pit".into(),
        games,
        opponents: vec!["scripted".into()],
        beta: 0.0,
        noise_prob: 0.0,
        noise_len: (2, 6),
        base_seed,
    }
}

/// A tiny MLP experiment over the `pit` arena with round-0 data collected into `tmp/base`, one DAgger
/// round with `dagger_jobs`.
fn tiny_experiment(tmp: &std::path::Path, dagger_jobs: Vec<JobConfig>) -> ExperimentConfig {
    let base = tmp.join("base");
    let dagger = tmp.join("dagger");
    let run_dir = tmp.join("run");

    // Round-0 data: two short planner games (a hold-out-free arena, so everything is training data
    // except the seed-divisible validation game).
    let env = load_env(
        &root().join("configs/arenas"),
        std::path::Path::new("/nonexistent"),
        None,
    )
    .unwrap();
    let _ = ModelBrains::new(None);
    let mut store = TeacherStore::create(&base, "base", "test").unwrap();
    let sums = run_collect_jobs(&env, &mut store, &[job(10, 3)], "teacher", 0, 2, &mut |_| {}).unwrap();
    assert_eq!(sums[0].games, 3);
    assert!(store.manifest.total_steps() > 30);

    ExperimentConfig {
        name: "smoke".into(),
        model: ModelSpec {
            kind: "mlp".into(),
            hidden: 3,
            lr: 5e-3,
        },
        flyg: "unused-by-controls".into(),
        brain_config: root().join("configs/fly/S-brain.toml").to_string_lossy().into_owned(),
        arenas_dir: root().join("configs/arenas").to_string_lossy().into_owned(),
        map_dir: "/nonexistent".into(),
        run_dir: run_dir.to_string_lossy().into_owned(),
        teacher_base: vec![base.to_string_lossy().into_owned()],
        teacher_dagger: dagger.to_string_lossy().into_owned(),
        train: TrainConfig {
            batch_windows: 4,
            window_len: 12,
            burn_in: 2,
            warmup_steps: 2,
            threads: 2,
            log_every: 2,
            eval_windows: 8,
            ..TrainConfig::default()
        },
        fly: Default::default(),
        teacher_data: TeacherDataConfig {
            val_mod: 3,
            ..TeacherDataConfig::default()
        },
        human: None,
        bc_steps: 6,
        eval_every: 0,
        dagger: DaggerConfig {
            betas: vec![0.5],
            noise_prob: 0.02,
            steps_per_round: 4,
            jobs: dagger_jobs,
            eval_games: 4,
            eval_arenas: vec!["pit".into()],
            retrain_lr_scale: 0.5,
        },
    }
}

#[test]
fn bc_then_a_dagger_round_writes_a_run_directory_and_resumes_as_a_no_op() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_experiment(tmp.path(), vec![job(500, 2)]);
    let dagger = tmp.path().join("dagger");
    let run_dir = tmp.path().join("run");
    let mut log = Vec::new();
    run_experiment(&cfg, &mut |l| log.push(l.to_string())).unwrap();

    for f in [
        "config.toml",
        "metrics.jsonl",
        "status.json",
        "state.bin",
        "checkpoints/last.bundle",
        "checkpoints/final.bundle",
        "rounds/round-0.bundle",
        "rounds/round-1.bundle",
    ] {
        assert!(run_dir.join(f).exists(), "missing {f}");
    }
    let metrics = std::fs::read_to_string(run_dir.join("metrics.jsonl")).unwrap();
    for kind in [
        "\"kind\":\"train\"",
        "\"kind\":\"eval\"",
        "\"kind\":\"collect\"",
        "\"kind\":\"arena\"",
    ] {
        assert!(metrics.contains(kind), "no {kind} line");
    }
    assert!(metrics.contains("dagger-1") && metrics.contains("\"phase\":\"bc\""));
    let ds = TeacherStore::open(&dagger).unwrap();
    assert_eq!(
        ds.chunks_of_round(Some(1)).len(),
        ds.manifest.chunks.len(),
        "the DAgger store holds round 1"
    );
    assert!(
        ds.manifest.chunks[0].actor.starts_with("mlp:"),
        "the student played round 1: {}",
        ds.manifest.chunks[0].actor
    );
    ds.verify().unwrap();
    let status: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("status.json")).unwrap()).unwrap();
    assert_eq!(status["phase"], "done");
    assert_eq!(status["step"], 10);

    // Resuming a finished run trains and collects nothing more.
    let chunks_before = ds.manifest.chunks.len();
    let lines_before = metrics.lines().count();
    let mut log2 = Vec::new();
    run_experiment(&cfg, &mut |l| log2.push(l.to_string())).unwrap();
    assert!(log2.iter().any(|l| l.contains("resumed at step 10")), "{log2:?}");
    assert_eq!(
        TeacherStore::open(&dagger).unwrap().manifest.chunks.len(),
        chunks_before
    );
    assert_eq!(
        std::fs::read_to_string(run_dir.join("metrics.jsonl"))
            .unwrap()
            .lines()
            .count(),
        lines_before
    );
}

/// Review F3 of E-005: a run killed between the jobs of a DAgger round must collect only the jobs
/// that are missing and train on the whole round, not silently on the part that was written.
#[test]
fn a_run_killed_between_collection_jobs_collects_only_the_missing_ones_on_resume() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_experiment(tmp.path(), vec![job(500, 2), job(600, 2)]);
    let dagger = tmp.path().join("dagger");
    let run_dir = tmp.path().join("run");

    // The kill: the log callback runs right after a job's chunks are in the manifest, so panicking
    // on the first "collected" line leaves exactly job 1 of 2 on disk.
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run_experiment(&cfg, &mut |l| {
            if l.starts_with("collected ") {
                panic!("simulated kill after the first job");
            }
        });
    }));
    assert!(killed.is_err(), "the run was killed");
    let partial = TeacherStore::open(&dagger).unwrap();
    assert!(!partial.chunks_of_round(Some(1)).is_empty(), "job 1 is on disk");
    assert!(
        !partial.manifest.round_complete(1),
        "but the round is not marked complete"
    );
    let keys_before: Vec<String> = partial.manifest.chunks.iter().map(|c| c.job_key.clone()).collect();
    assert_eq!(keys_before.len(), 1, "exactly one job's chunks: {keys_before:?}");
    assert!(keys_before[0].contains("seed=100000500"), "{keys_before:?}");

    // Resume: the second job is collected, the first is kept as it is, and training sees both.
    let mut log = Vec::new();
    run_experiment(&cfg, &mut |l| log.push(l.to_string())).unwrap();
    assert!(
        log.iter()
            .any(|l| l.contains("resuming collection, 1 of 2 jobs already collected")),
        "{log:?}"
    );
    let done = TeacherStore::open(&dagger).unwrap();
    assert!(done.manifest.round_complete(1));
    let mut keys: Vec<String> = done
        .manifest
        .chunks
        .iter()
        .filter(|c| c.round == 1)
        .map(|c| c.job_key.clone())
        .collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 2, "both jobs, each once: {keys:?}");
    assert!(
        keys[0].contains("seed=100000500") && keys[1].contains("seed=100000600"),
        "{keys:?}"
    );
    assert_eq!(
        done.chunks_of_round(Some(1)).len(),
        done.manifest.chunks.len(),
        "no job was collected twice"
    );
    // The first job's chunk is the one written before the kill (kept, not redone).
    assert_eq!(done.manifest.chunks[0], partial.manifest.chunks[0]);
    done.verify().unwrap();
    let metrics = std::fs::read_to_string(run_dir.join("metrics.jsonl")).unwrap();
    assert!(
        metrics.contains("\"jobs_resumed\":1"),
        "the collect line says it resumed"
    );
    assert!(
        metrics.contains("\"kind\":\"hook_play\""),
        "the in-play hook split is logged per round"
    );
    let status: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run_dir.join("status.json")).unwrap()).unwrap();
    assert_eq!(status["phase"], "done");

    // And a finished round is never collected again.
    let chunks = done.manifest.chunks.len();
    run_experiment(&cfg, &mut |_| {}).unwrap();
    assert_eq!(TeacherStore::open(&dagger).unwrap().manifest.chunks.len(), chunks);
}
