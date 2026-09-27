//! Acceptance criterion 4's throughput measurement: sample-steps/s (one "sample-step" = one
//! decision of one sequence including its substeps, forward+backward) for `B = 64`, `T = 32`
//! decisions, on the real S/M graphs, compared with the phase-0 prediction (`docs/research/
//! rust-stack.md` §4: "≈13-18k sample-steps/s at 0.5M synapses on 8 threads; ~43k at 0.2M").
//!
//! `#[ignore]`d (needs real `.flyg` data — see `tests/stability.rs`'s doc comment for the
//! project's standard convention). Run with:
//! `cargo test -p ddai-fly --release --test train_throughput -- --ignored --nocapture`
//!
//! Thread counts and machine load: the harness this ran under shares the host with other agents'
//! builds/tests, so it always measures at whatever thread count is passed on the command line
//! (default: 1 and 6 — see `CLAUDE.md`/the task's own instruction to cap rayon benchmarks at 6
//! threads while other builds may be running) and always prints `/proc/loadavg` right before each
//! timed run, so a report quoting these numbers can show the contention they were taken under. Set
//! `FLY_BENCH_THREADS` (comma-separated, e.g. `1,8`) to override.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ddai_fly::rng::SplitMix64;
use ddai_fly::{BackwardIndex, FlyConfig, FlyModel, FlyParams, FlyState, Sequence, train_step};

fn compiled_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set to find ~/aiddnet/data/connectome/compiled");
    PathBuf::from(home).join("aiddnet/data/connectome/compiled")
}

fn read_load_average() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .take(3)
                .map(str::to_string)
                .reduce(|a, b| format!("{a} {b}"))
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

/// Peak resident set size in KiB (`VmHWM` from `/proc/self/status`) — Linux-specific, matches this
/// project's other `/proc`-reading conventions (`ddnet-ai fly bench`'s own load-average read).
/// `None` on a platform/sandbox without `/proc` (never a hard failure — just an unreported number).
fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

fn thread_counts() -> Vec<usize> {
    if let Ok(s) = std::env::var("FLY_BENCH_THREADS") {
        return s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    }
    vec![1, 6]
}

fn build_sequences(model: &FlyModel, rng: &mut SplitMix64, batch: usize, t_decisions: usize) -> Vec<Sequence> {
    let num_inputs = model.num_inputs();
    let num_outputs = model.num_outputs();
    (0..batch)
        .map(|_| {
            let v_init: Vec<f32> = (0..model.num_neurons())
                .map(|_| rng.next_f32_unit() * 0.4 - 0.2)
                .collect();
            let inputs: Vec<Vec<f32>> = (0..t_decisions)
                .map(|_| (0..num_inputs).map(|_| rng.next_f32_unit() * 0.6 - 0.3).collect())
                .collect();
            let grad_dn: Vec<Vec<f32>> = (0..t_decisions)
                .map(|_| (0..num_outputs).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect())
                .collect();
            Sequence {
                v_init,
                inputs,
                grad_dn,
                extra_taps: vec![],
            }
        })
        .collect()
}

fn bench_one_graph(name: &str, path: &Path) {
    if !path.exists() {
        eprintln!(
            "train_throughput: skipping {name} ({} not found in this environment)",
            path.display()
        );
        return;
    }
    let flyg = ddai_flyg::load(path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    let config = FlyConfig::default();
    let params = FlyParams::init_default(&flyg, &config, 42);
    let model = FlyModel::new(flyg, config, params).expect("build model");
    let index = BackwardIndex::build(&model);

    // Realistic-ish starting states: warm the model up once and perturb from there per sequence
    // (build_sequences already draws v_init independently per sequence, which is a reasonable
    // stand-in for "many different episodes" without needing a real warm-up per sequence in this
    // throughput measurement).
    let mut warm = FlyState::new(&model);
    let _ = warm.warm_up(&model);

    let mut rng = SplitMix64::new(20260927);
    const BATCH: usize = 64;
    const T_DECISIONS: usize = 32;
    let sequences = build_sequences(&model, &mut rng, BATCH, T_DECISIONS);
    let substeps = model.config().substeps_per_decision as usize;

    eprintln!(
        "\n=== {name}: {} neurons, {} edges, B={BATCH}, T={T_DECISIONS}, substeps={substeps} ===",
        model.num_neurons(),
        model.flyg().edges.num_edges()
    );

    for &threads in &thread_counts() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build thread pool");

        // Warm-up run (not timed): first-touch page faults / branch predictor warm-up shouldn't
        // count against the measured throughput.
        pool.install(|| {
            let _ = train_step(&model, &index, &sequences, None).unwrap();
        });

        let load_before = read_load_average();
        const REPEATS: u32 = 5;
        let start = Instant::now();
        for _ in 0..REPEATS {
            pool.install(|| {
                let _ = train_step(&model, &index, &sequences, None).unwrap();
            });
        }
        let elapsed = start.elapsed();
        let total_sample_steps = (BATCH * T_DECISIONS * REPEATS as usize) as f64;
        let sample_steps_per_sec = total_sample_steps / elapsed.as_secs_f64();
        let peak_rss = peak_rss_kb();

        eprintln!(
            "  threads={threads:>2}  load_avg_before={load_before:<18}  elapsed={:>8.3?} for {REPEATS} repeats  \
             sample-steps/s={sample_steps_per_sec:>10.0}  peak_rss={}",
            elapsed / REPEATS,
            peak_rss
                .map(|kb| format!("{:.1} MiB", kb as f64 / 1024.0))
                .unwrap_or_else(|| "n/a".to_string()),
        );
    }
}

#[test]
#[ignore]
fn throughput_on_real_s_and_m_graphs() {
    let base = compiled_dir();
    bench_one_graph("S", &base.join("fly-S-v1.flyg"));
    bench_one_graph("M", &base.join("fly-M-v1.flyg"));
}

/// Runs the exact same `build_sequences` + `train_step` wiring `bench_one_graph` uses, but on a
/// tiny offline fixture graph (in normal, non-`#[ignore]`d CI) — not a throughput measurement (far
/// too small to mean anything), just a guard so a regression in that wiring is caught immediately
/// rather than only when someone runs the `#[ignore]`d real-graph benchmark.
#[test]
fn throughput_harness_wiring_runs_end_to_end_on_a_tiny_graph() {
    use ddai_fly::test_fixtures::tiny_chain_flyg;

    let flyg = tiny_chain_flyg();
    let config = FlyConfig {
        substeps_per_decision: 2,
        ..FlyConfig::default()
    };
    let params = FlyParams::init_default(&flyg, &config, 1);
    let model = FlyModel::new(flyg, config, params).unwrap();
    let index = BackwardIndex::build(&model);

    let mut rng = SplitMix64::new(7);
    let sequences = build_sequences(&model, &mut rng, 4, 3);
    assert_eq!(sequences.len(), 4);

    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let batch = pool.install(|| train_step(&model, &index, &sequences, None).unwrap());
    assert!(batch.grad.all_finite());
    assert_eq!(batch.grad_inputs.len(), 4);
}
